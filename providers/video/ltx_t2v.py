"""LTX-2.5 text-to-video provider.

Uses the verified template workflows/video/ltx2_5_t2v.json, derived from
the I2V graph (see export_ltx_t2v.py): identical loaders, guiders,
samplers, duration math and resolution handling, but NO image
conditioning -- the two LTXVConcatAVLatent video_latent inputs sit
directly on latents, so no input image is uploaded, required, or read.

Duration convention (same as I2V): the duration PrimitiveInt holds whole
seconds; ComfyMathExpression computes frames = seconds * 24 + 1, i.e. a
24 fps timeline (PrimitiveInt 1361). A 7s storyboard scene therefore
renders 169 frames (~7.04s of video).

Only these inputs are ever substituted: positive prompt, negative prompt,
seed, duration/frame count, and the ResolutionSelector. Nothing else in
the graph is touched.
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

from ..comfy import ComfyClient, ComfyError
from ..errors import ProviderError
from .base import VideoRequest, VideoResult
from .ltx import LENGTH_MATH_EXPRESSION, LtxVideoProvider

FRAME_RATE = 24.0

REQUIRED_NODE_CLASSES = (
    "EmptyLTXVLatentVideo",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "SamplerCustomAdvanced",
    "ResolutionSelector",
)


class LtxT2VVideoProvider(LtxVideoProvider):
    provider_id = "ltx_t2v"

    # Pipeline contract: this backend takes no input image.
    needs_input_image = False

    def __init__(self, name: str = "ltx_t2v",
                 settings: dict | None = None,
                 client: ComfyClient | None = None):
        # Bypass LtxVideoProvider.__init__ only in name; settings/client
        # handling is identical.
        super().__init__(name=name, settings=settings, client=client)

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
                self.provider_id, f"Workflow is not the LTX-2.5 T2V graph; missing: {missing}.",
                workflow=os.path.basename(self.workflow_path))
        for banned in ("LoadImage", "LTXVPreprocess", "LTXVImgToVideoInplace",
                       "TextGenerateLTX2Prompt"):
            if banned in present:
                raise ProviderError(
                    self.provider_id,
                    f"Workflow is not text-to-video; image conditioning node present: {banned}.",
                    workflow=os.path.basename(self.workflow_path))
        return wf

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)

        prompts = by_class.get("PrimitiveStringMultiline", [])
        if len(prompts) != 1:
            raise ProviderError("ltx_t2v", f"Expected 1 prompt node, found {prompts}.")
        negatives = [nid for nid in by_class.get("CLIPTextEncode", [])
                     if isinstance(workflow[nid]["inputs"].get("text"), str)]
        if len(negatives) != 1:
            raise ProviderError("ltx_t2v", f"Expected 1 literal-text negative node, found {negatives}.")
        seeds = by_class.get("RandomNoise", [])
        if not seeds:
            raise ProviderError("ltx_t2v", "No RandomNoise nodes found.")
        duration_id = None
        for nid in by_class.get("ComfyMathExpression", []):
            if workflow[nid]["inputs"].get("expression") == LENGTH_MATH_EXPRESSION:
                src = workflow[nid]["inputs"].get("values.a")
                if (isinstance(src, list)
                        and workflow.get(str(src[0]), {}).get("class_type") == "PrimitiveInt"):
                    duration_id = str(src[0])
        if duration_id is None:
            raise ProviderError("ltx_t2v", "Duration PrimitiveInt not found.")
        if by_class.get("LoadImage"):
            raise ProviderError("ltx_t2v", "Text-to-video graph must not contain LoadImage.")
        selectors = by_class.get("ResolutionSelector", [])
        if len(selectors) != 1:
            raise ProviderError("ltx_t2v", f"Expected 1 ResolutionSelector node, found {selectors}.")
        return {"prompt": prompts[0], "negative": negatives[0], "seeds": seeds,
                "duration": duration_id, "resolution": selectors[0]}

    def customize(self, template: dict, *, video_prompt: str, negative_prompt: str,
                  seed: int, duration: int,
                  aspect_label: str, megapixels: float) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["prompt"]]["inputs"]["value"] = video_prompt
        wf[found["negative"]]["inputs"]["text"] = negative_prompt
        for nid in found["seeds"]:
            wf[nid]["inputs"]["noise_seed"] = seed
        wf[found["duration"]]["inputs"]["value"] = duration
        wf[found["resolution"]]["inputs"] = {
            "aspect_ratio": aspect_label, "megapixels": megapixels, "multiple": 32}
        return wf

    # -- VideoProvider interface --
    def generate(self, request: VideoRequest) -> VideoResult:
        if request.input_image is not None:
            raise ProviderError(self.provider_id,
                                "Text-to-video takes no input image; got "
                                f"{request.input_image}.")
        duration = int(round(request.duration))
        if duration < 1 or duration > 30:
            raise ValueError(f"Duration out of range (1..30s): {request.duration}.")
        width, height = self.adjust_dimensions(request.width, request.height)
        aspect_label, megapixels, (final_w, final_h) = self.selector_settings(width, height)
        seed = request.seed if request.seed is not None else random.randint(0, 2**31 - 1)
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            wf = self.customize(template, video_prompt=request.prompt,
                                negative_prompt=request.negative_prompt, seed=seed,
                                duration=duration,
                                aspect_label=aspect_label, megapixels=megapixels)
            dest = self.client.run(wf, ("video", "gifs", "images"), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=final_w, height=final_h)
