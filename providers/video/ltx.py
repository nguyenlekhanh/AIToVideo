"""LTX-2.5 image-to-video provider.

Uses the verified template export workflows/video/ltx2_5_i2v.json EXACTLY:
EmptyLTXVLatentVideo + LTXVPreprocess + LTXVImgToVideoInplace guides +
LTXVConcatAVLatent + LTXVSeparateAVLatent + SamplerCustomAdvanced (base +
x2 spatial upscale), LTX-2.5 transformer/VAEs/CLIPs.

Only these inputs are ever substituted: positive prompt, negative prompt,
seed, duration/frame count, input image, and the ResolutionSelector
(requested dimensions). Nothing else in the graph is touched.
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

LENGTH_MATH_EXPRESSION = "a * b + 1"
DIM_MULTIPLE = 32

REQUIRED_NODE_CLASSES = (
    "EmptyLTXVLatentVideo",
    "LTXVPreprocess",
    "LTXVImgToVideoInplace",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "SamplerCustomAdvanced",
    "LoadImage",
)


class LtxVideoProvider(VideoProvider):
    provider_id = "ltx"

    def __init__(self, name: str = "ltx", settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))

    # -- dimensions: LTX requires multiples of 32; the template's
    # ResolutionSelector recomputes output size from aspect + megapixels,
    # so the provider predicts that result and reports it. --
    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return snap(width, DIM_MULTIPLE), snap(height, DIM_MULTIPLE)

    def selector_settings(self, width: int, height: int) -> tuple[str, float, int]:
        """Map requested dims to ResolutionSelector (aspect label, MP, multiple)
        plus the expected output dims.

        The selector derives height from aspect + megapixels and then width
        from the snapped height (both snapped to 32). This height-first
        formula matches both verified renders: 3:2 @ 0.4MP -> 768x512 and
        9:16 @ 1.0MP -> 768x1344.
        """
        width, height = self.adjust_dimensions(width, height)
        label, aw, ah = select_aspect_label(width, height)
        megapixels = max(0.1, round(width * height / 1_000_000, 1))
        unit = math.sqrt(megapixels * 1_000_000 / (aw * ah))
        final_h = snap(round(ah * unit), DIM_MULTIPLE)
        final_w = snap(round(final_h * aw / ah), DIM_MULTIPLE)
        return label, megapixels, (final_w, final_h)

    # -- workflow handling (structure-aware, id-independent) --
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
                self.provider_id, f"Workflow is not the LTX-2.5 I2V graph; missing: {missing}.",
                workflow=os.path.basename(self.workflow_path))
        return wf

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)

        prompts = by_class.get("PrimitiveStringMultiline", [])
        if len(prompts) != 1:
            raise ProviderError("ltx", f"Expected 1 prompt node, found {prompts}.")
        negatives = [nid for nid in by_class.get("CLIPTextEncode", [])
                     if isinstance(workflow[nid]["inputs"].get("text"), str)]
        if len(negatives) != 1:
            raise ProviderError("ltx", f"Expected 1 literal-text negative node, found {negatives}.")
        seeds = by_class.get("RandomNoise", [])
        if not seeds:
            raise ProviderError("ltx", "No RandomNoise nodes found.")
        duration_id = None
        for nid in by_class.get("ComfyMathExpression", []):
            if workflow[nid]["inputs"].get("expression") == LENGTH_MATH_EXPRESSION:
                src = workflow[nid]["inputs"].get("values.a")
                if (isinstance(src, list)
                        and workflow.get(str(src[0]), {}).get("class_type") == "PrimitiveInt"):
                    duration_id = str(src[0])
        if duration_id is None:
            raise ProviderError("ltx", "Duration PrimitiveInt not found.")
        images = by_class.get("LoadImage", [])
        if len(images) != 1:
            raise ProviderError("ltx", f"Expected 1 LoadImage node, found {images}.")
        selectors = by_class.get("ResolutionSelector", [])
        if len(selectors) != 1:
            raise ProviderError("ltx", f"Expected 1 ResolutionSelector node, found {selectors}.")
        return {"prompt": prompts[0], "negative": negatives[0], "seeds": seeds,
                "duration": duration_id, "image": images[0],
                "resolution": selectors[0]}

    def customize(self, template: dict, *, video_prompt: str, negative_prompt: str,
                  seed: int, duration: int, image_name: str,
                  aspect_label: str, megapixels: float) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["prompt"]]["inputs"]["value"] = video_prompt
        wf[found["negative"]]["inputs"]["text"] = negative_prompt
        for nid in found["seeds"]:
            wf[nid]["inputs"]["noise_seed"] = seed
        wf[found["duration"]]["inputs"]["value"] = duration
        wf[found["image"]]["inputs"]["image"] = image_name
        wf[found["resolution"]]["inputs"] = {
            "aspect_ratio": aspect_label, "megapixels": megapixels, "multiple": 32}
        return wf

    # -- VideoProvider interface --
    def generate(self, request: VideoRequest) -> VideoResult:
        if request.input_image is not None and not os.path.isfile(request.input_image):
            raise ProviderError(self.provider_id,
                                f"Input image not found: {request.input_image}.")
        duration = int(round(request.duration))
        if duration < 1 or duration > 30:
            raise ValueError(f"Duration out of range (1..30s): {request.duration}.")
        width, height = self.adjust_dimensions(request.width, request.height)
        aspect_label, megapixels, (final_w, final_h) = self.selector_settings(width, height)
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            stored = self.client.upload_image(str(request.input_image))
            wf = self.customize(template, video_prompt=request.prompt,
                                negative_prompt=request.negative_prompt, seed=seed,
                                duration=duration, image_name=stored,
                                aspect_label=aspect_label, megapixels=megapixels)
            dest = self.client.run(wf, ("video", "gifs", "images"), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=final_w, height=final_h)
