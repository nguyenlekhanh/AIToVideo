"""MiniMax H3 reference-to-video provider (R2V + memory optimize + sparse attention).

Uses the verified template workflows/video/minimax_h3_r2v.json, flattened
from workflows_template/video_minimax_h3_r2v_with_H3_Memory_Optimize_H3_Sparse_Attention.json
(see export_minimax_h3_r2v.py). Preserves H3MemoryOptimization (x2) and
H3SparseAttention (x2) nodes verbatim.

Reference images: up to 4 product/character images -> ref_images.ref_image_0..3
via the template's four LoadImage nodes (139/147/137/153). Product references
are placed before character references; the two concepts remain distinct in
VideoRequest. Guide frame (initial image or previous clip's last frame) ->
ref_images.ref_image_4 via an in-memory LoadImage node "1" (optional slot,
shape 7, null by default) and is never used as a product reference.

Duration convention (same as T2V): PrimitiveFloat 132 holds seconds;
ComfyMathExpression 131 snaps to 17k+5 lattice at 24 fps.
Resolution via ResolutionSelector 115 (links 276/277).

Only these inputs are ever substituted: prompt (138), width/height (136
via 115), duration seconds (132), seed (129), character refs (LoadImage
139/147/137/153), guide frame (in-memory LoadImage "1" -> ref_image_4).
Nothing else in the graph is touched (Memory Optimize / Sparse Attention
config untouched).
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
MAX_DURATION = 15

REQUIRED_NODE_CLASSES = (
    "MiniMaxH3ReferenceToVideo",
    "RandomNoise",
    "PrimitiveFloat",
    "ResolutionSelector",
    "SaveVideo",
    "CreateVideo",
    "H3MemoryOptimization",
    "H3SparseAttention",
)

GUIDE_NODE_ID = "1"
MAX_CHAR_REFS = 4


class MiniMaxH3R2VProvider(VideoProvider):
    provider_id = "minimax_h3_r2v"
    needs_input_image = False
    supports_first_frame = True
    supports_character_refs = True
    supports_product_refs = True

    def __init__(self, name: str = "minimax_h3_r2v",
                 settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))
        self._upload_cache: dict[str, str] = {}

    def _upload(self, path: str) -> str:
        if path in self._upload_cache:
            return self._upload_cache[path]
        name = self.client.upload_image(path)
        self._upload_cache[path] = name
        return name

    def adjust_dimensions(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return snap(width, DIM_MULTIPLE), snap(height, DIM_MULTIPLE)

    def selector_settings(self, width, height):
        width, height = self.adjust_dimensions(width, height)
        label, aw, ah = select_aspect_label(width, height)
        megapixels = max(0.1, round(width * height / 1_000_000, 1))
        unit = math.sqrt(megapixels * 1_000_000 / (aw * ah))
        final_h = snap(round(ah * unit), DIM_MULTIPLE)
        final_w = snap(round(final_h * aw / ah), DIM_MULTIPLE)
        return label, megapixels, (final_w, final_h)

    def load_workflow(self):
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
                self.provider_id, f"Workflow is not the MiniMax H3 R2V graph; missing: {missing}.",
                workflow=os.path.basename(self.workflow_path))
        return wf

    @staticmethod
    def _find_nodes(workflow):
        by_class = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)
        gens = by_class.get("MiniMaxH3ReferenceToVideo", [])
        if len(gens) != 1:
            raise ProviderError("minimax_h3_r2v",
                                f"Expected 1 MiniMaxH3ReferenceToVideo node, found {gens}.")
        seeds = by_class.get("RandomNoise", [])
        if len(seeds) != 1:
            raise ProviderError("minimax_h3_r2v",
                                f"Expected 1 RandomNoise node, found {seeds}.")
        duration_id = None
        for nid in by_class.get("PrimitiveFloat", []):
            for other_id, other in workflow.items():
                if other.get("class_type") != "ComfyMathExpression":
                    continue
                for value in (other.get("inputs") or {}).values():
                    if isinstance(value, list) and str(value[0]) == nid:
                        duration_id = nid
        if duration_id is None:
            raise ProviderError("minimax_h3_r2v", "Duration PrimitiveFloat not found.")
        selectors = by_class.get("ResolutionSelector", [])
        if len(selectors) != 1:
            raise ProviderError("minimax_h3_r2v",
                                f"Expected 1 ResolutionSelector node, found {selectors}.")
        saves = by_class.get("SaveVideo", [])
        if len(saves) != 1:
            raise ProviderError("minimax_h3_r2v",
                                f"Expected 1 SaveVideo node, found {saves}.")
        prompts = by_class.get("PrimitiveStringMultiline", [])
        if len(prompts) != 1:
            raise ProviderError("minimax_h3_r2v",
                                f"Expected 1 prompt node, found {prompts}.")
        return {"generate": gens[0], "seed": seeds[0], "duration": duration_id,
                "resolution": selectors[0], "save": saves[0],
                "prompt": prompts[0]}

    def customize(self, template, *, video_prompt, seed, duration,
                  aspect_label, megapixels, character_refs=None,
                  product_refs=None, first_frame=None):
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["prompt"]]["inputs"]["value"] = video_prompt
        wf[found["seed"]]["inputs"]["noise_seed"] = seed
        wf[found["duration"]]["inputs"]["value"] = float(duration)
        wf[found["resolution"]]["inputs"] = {
            "aspect_ratio": aspect_label, "megapixels": megapixels, "multiple": 32}
        gen = wf[found["generate"]]["inputs"]
        # Product and character refs are distinct inputs but share the
        # template's four LoadImage slots. Product identity comes first so
        # the prompt's <Picture 1..k> tags remain stable across clips.
        # Slot origins are read from the canonical graph (0->137, 1->139,
        # 2->147, 3->153), so mapping always follows the template.
        slots = []
        for i in range(MAX_CHAR_REFS):
            key = f"ref_images.ref_image_{i}"
            link = gen.get(key)
            if not (isinstance(link, list) and link):
                break
            node_id = str(link[0])
            node = wf.get(node_id) or {}
            if node.get("class_type") != "LoadImage":
                raise ProviderError(
                    self.provider_id,
                    f"ref_image_{i} is not fed by a LoadImage node.")
            slots.append((key, node_id))
        product = list(product_refs or [])
        character = list(character_refs or [])
        refs = product + character
        if len(refs) > len(slots):
            raise ValueError(
                f"Too many reference images ({len(refs)}: "
                f"{len(product)} product + {len(character)} character); "
                f"workflow supports at most {len(slots)}.")
        uploaded = [self._upload(p) for p in refs]
        for i, (key, node_id) in enumerate(slots):
            if i < len(uploaded):
                wf[node_id]["inputs"]["image"] = uploaded[i]
            else:
                # disconnect unused ref slots
                gen[key] = None
        if first_frame is not None:
            # guide frame -> ref_image_4 via in-memory LoadImage "1"
            wf[GUIDE_NODE_ID] = {"class_type": "LoadImage",
                                 "inputs": {"image": first_frame}}
            gen["ref_images.ref_image_4"] = [GUIDE_NODE_ID, 0]
        return wf

    def generate(self, request):
        if request.input_image is not None:
            raise ProviderError(self.provider_id,
                                "Reference-to-video takes no scene input "
                                f"image; got {request.input_image}.")
        if (request.negative_prompt or "").strip():
            raise ProviderError(
                self.provider_id,
                "The MiniMax H3 R2V graph has no negative prompt input; "
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
                                character_refs=request.character_refs,
                                product_refs=request.product_refs,
                                first_frame=request.first_frame)
            dest = self.client.run(wf, ("video", "gifs", "images"), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=final_w, height=final_h)
