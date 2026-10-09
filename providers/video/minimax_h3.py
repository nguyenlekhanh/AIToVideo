"""MiniMax H3 text-to-video provider.

Uses the verified template workflows/video/minimax_h3_t2v.json, flattened
from workflows_template/video_minimax_h3_t2v.json (see
export_minimax_h3_template.py): loaders, turbo LoRA switch, scheduler,
duration math, resolution selector, CreateVideo (+native audio) and
SaveVideo. The subgraph's first_frame/last_frame inputs are unlinked in
the template and omitted here (server defaults apply = pure T2V).

Runtime node IDs (flat API graph; inner subgraph ids remapped +1000):
  - 1131 MiniMaxH3ImageToVideo: `prompt` injected per scene; `width`/
    `height` linked from ResolutionSelector 115 slots 0/1 (template links
    246/247); `length` from duration math. `first_frame`/`last_frame`
    intentionally absent (NOT used, NOT wired).
  - 1129 RandomNoise: `noise_seed` injected per scene.
  - 1133 PrimitiveFloat: `value` = duration seconds (float).
  - 1132 ComfyMathExpression: snaps seconds to the 17k+5 frame lattice.
  - 115 ResolutionSelector: `aspect_ratio`/`megapixels` injected per scene.
  - 1130 CreateVideo (images + native audio) -> 92 SaveVideo.

  Optional first-frame chaining (--use-first-frame): when a
  `first_frame` upload name is supplied, a LoadImage node "1"
  ({"image": <uploaded name>}, the existing ComfyUI upload mechanism)
  is added in memory and 1131 `first_frame` is linked to ["1", 0]
  (template inner link 195, subgraph slot 0). `last_frame` is NEVER
  populated. Without it, the graph is pure T2V as above.

Duration convention (from the template, NOT copied from LTX): the
duration PrimitiveFloat holds whole seconds; ComfyMathExpression snaps
to the model's 17k+5 frame lattice at 24 fps
(max(5, round(s*24)) + (5 - (... % 17)) % 17). E.g. 5s -> 124 frames
(~5.17s), 7s -> 175 frames (~7.29s). Trained range is ~124-362 frames
(~5-15s); durations outside 1..15s are rejected rather than faked.

There is no negative-prompt path in this graph: a non-empty
negative_prompt is refused loudly (Krea precedent).

Only these inputs are ever substituted: positive prompt, seed,
duration seconds, and the ResolutionSelector (requested dimensions).
Nothing else in the graph is touched.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path

from ..comfy import ComfyClient, ComfyError
from ..dims import select_aspect_label, snap
from ..errors import ProviderError
from .base import VideoProvider, VideoRequest, VideoResult

DIM_MULTIPLE = 32
MIN_DURATION = 1
MAX_DURATION = 15  # template math is trained/tested up to ~362 frames (~15s)

REQUIRED_NODE_CLASSES = (
    "MiniMaxH3ImageToVideo",
    "RandomNoise",
    "PrimitiveFloat",
    "ResolutionSelector",
    "SaveVideo",
    "CreateVideo",
)


class MiniMaxH3VideoProvider(VideoProvider):
    provider_id = "minimax_h3"

    # Pipeline contract: this backend takes no input image.
    needs_input_image = False
    # Optional continuity chaining (--use-first-frame): an uploaded PNG
    # may feed the first_frame input for clips after the first.
    supports_first_frame = True
    # In-memory LoadImage id for chaining (free: runtime ids are
    # 92/115/1119-1139). Never written to the template on disk.
    FIRST_FRAME_NODE_ID = "1"

    def __init__(self, name: str = "minimax_h3",
                 settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))

    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return snap(width, DIM_MULTIPLE), snap(height, DIM_MULTIPLE)

    def selector_settings(self, width: int, height: int) -> tuple[str, float, int]:
        """Same height-first ResolutionSelector mapping as the LTX provider."""
        width, height = self.adjust_dimensions(width, height)
        label, aw, ah = select_aspect_label(width, height)
        megapixels = max(0.1, round(width * height / 1_000_000, 1))
        unit = math.sqrt(megapixels * 1_000_000 / (aw * ah))
        final_h = snap(round(ah * unit), DIM_MULTIPLE)
        final_w = snap(round(final_h * aw / ah), DIM_MULTIPLE)
        return label, megapixels, (final_w, final_h)

    def load_workflow(self) -> dict:
        if not self.workflow_path or not os.path.isfile(self.workflow_path):
            raise ProviderError(self.provider_id, "Workflow file not found.",
                                workflow=self.workflow_path or "(unset)")
        with open(self.workflow_path, encoding="utf-8") as f:
            wf = json.load(f)
        wf.pop("_meta", None)
        present = {n.get("class_type") for n in wf.values()}
        missing = [c for c in REQUIRED_NODE_CLASSES if c not in present]
        if missing:
            raise ProviderError(
                self.provider_id, f"Workflow is not the MiniMax H3 T2V graph; missing: {missing}.",
                workflow=os.path.basename(self.workflow_path))
        if "LoadImage" in present:
            raise ProviderError(
                self.provider_id,
                "Workflow is not text-to-video; image conditioning node present: LoadImage.",
                workflow=os.path.basename(self.workflow_path))
        return wf

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)

        gens = by_class.get("MiniMaxH3ImageToVideo", [])
        if len(gens) != 1:
            raise ProviderError("minimax_h3",
                                f"Expected 1 MiniMaxH3ImageToVideo node, found {gens}.")
        seeds = by_class.get("RandomNoise", [])
        if len(seeds) != 1:
            raise ProviderError("minimax_h3",
                                f"Expected 1 RandomNoise node, found {seeds}.")
        # Duration node: the PrimitiveFloat feeding a ComfyMathExpression.
        duration_id = None
        for nid in by_class.get("PrimitiveFloat", []):
            for other_id, other in workflow.items():
                if other.get("class_type") != "ComfyMathExpression":
                    continue
                for value in (other.get("inputs") or {}).values():
                    if isinstance(value, list) and str(value[0]) == nid:
                        duration_id = nid
        if duration_id is None:
            raise ProviderError("minimax_h3", "Duration PrimitiveFloat not found.")
        selectors = by_class.get("ResolutionSelector", [])
        if len(selectors) != 1:
            raise ProviderError("minimax_h3",
                                f"Expected 1 ResolutionSelector node, found {selectors}.")
        saves = by_class.get("SaveVideo", [])
        if len(saves) != 1:
            raise ProviderError("minimax_h3",
                                f"Expected 1 SaveVideo node, found {saves}.")
        return {"generate": gens[0], "seed": seeds[0], "duration": duration_id,
                "resolution": selectors[0], "save": saves[0]}

    def customize(self, template: dict, *, video_prompt: str, seed: int,
                  duration: float, aspect_label: str,
                  megapixels: float,
                  first_frame: str | None = None) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["generate"]]["inputs"]["prompt"] = video_prompt
        wf[found["seed"]]["inputs"]["noise_seed"] = seed
        wf[found["duration"]]["inputs"]["value"] = float(duration)
        wf[found["resolution"]]["inputs"] = {
            "aspect_ratio": aspect_label, "megapixels": megapixels, "multiple": 32}
        if first_frame is not None:
            # Continuity chaining: uploaded PNG -> LoadImage -> generate
            # first_frame (template inner link 195). last_frame untouched.
            wf[self.FIRST_FRAME_NODE_ID] = {
                "class_type": "LoadImage",
                "inputs": {"image": first_frame}}
            wf[found["generate"]]["inputs"]["first_frame"] = [
                self.FIRST_FRAME_NODE_ID, 0]
        return wf

    # -- VideoProvider interface --
    def generate(self, request: VideoRequest) -> VideoResult:
        if request.input_image is not None:
            raise ProviderError(self.provider_id,
                                "Text-to-video takes no input image; got "
                                f"{request.input_image}.")
        if (request.negative_prompt or "").strip():
            raise ProviderError(
                self.provider_id,
                "The MiniMax H3 graph has no negative prompt input; "
                "refusing non-empty negative_prompt.")
        duration = float(request.duration)
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            raise ValueError(
                f"Duration out of range ({MIN_DURATION}..{MAX_DURATION}s): "
                f"{request.duration}.")
        width, height = self.adjust_dimensions(request.width, request.height)
        aspect_label, megapixels, (final_w, final_h) = self.selector_settings(width, height)
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            wf = self.customize(template, video_prompt=request.prompt,
                                seed=seed, duration=duration,
                                aspect_label=aspect_label, megapixels=megapixels,
                                first_frame=request.first_frame)
            dest = self.client.run(wf, ("video", "gifs", "images"), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=final_w, height=final_h)
