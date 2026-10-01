"""Wan2.1 image-to-video provider (480p, 16 fps).

Graph: UNETLoader (Wan2.1 I2V) + UMT5 CLIPLoader + CLIP-Vision +
Wan VAE + WanImageToVideo + KSampler + VAEDecode + VHS_VideoCombine.
Only prompt / negative / seed / frames / dims / image are substituted.
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

from ..comfy import ComfyClient, ComfyError
from ..dims import snap
from ..errors import ProviderError
from .base import VideoProvider, VideoRequest, VideoResult

DIM_MULTIPLE = 16
FRAME_RATE = 16.0


class WanVideoProvider(VideoProvider):
    provider_id = "wan"

    def __init__(self, name: str = "wan", settings: dict | None = None,
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

    @staticmethod
    def frames_for_duration(duration: float) -> int:
        """Wan length is 4k+1 frames."""
        total = max(1, int(round(duration * FRAME_RATE)))
        return (total // 4) * 4 + 1

    def load_workflow(self) -> dict:
        if not self.workflow_path or not os.path.isfile(self.workflow_path):
            raise ProviderError(self.provider_id, "Workflow file not found.",
                                workflow=self.workflow_path or "(unset)")
        with open(self.workflow_path, encoding="utf-8") as f:
            wf = json.load(f)
        wf.pop("_meta", None)
        return wf

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)
        for cls in ("WanImageToVideo", "KSampler", "LoadImage"):
            if len(by_class.get(cls, [])) != 1:
                raise ProviderError("wan",
                                    f"Expected 1x {cls}, found {by_class.get(cls, [])}.")
        sampler_inputs = workflow[by_class["KSampler"][0]]["inputs"]
        i2v_id = by_class["WanImageToVideo"][0]
        i2v_inputs = workflow[i2v_id]["inputs"]
        try:
            pos_id = str(i2v_inputs["positive"][0])
            neg_id = str(i2v_inputs["negative"][0])
        except (KeyError, TypeError, IndexError):
            raise ProviderError("wan", "WanImageToVideo positive/negative links not found.")
        if workflow[pos_id].get("class_type") != "CLIPTextEncode":
            raise ProviderError("wan", "Wan positive input is not CLIPTextEncode.")
        if workflow[neg_id].get("class_type") != "CLIPTextEncode":
            raise ProviderError("wan", "Wan negative input is not CLIPTextEncode.")
        _ = sampler_inputs  # sampler wiring (model/conditioning/latent) is template-fixed.
        return {"positive": pos_id, "negative": neg_id,
                "sampler": by_class["KSampler"][0],
                "i2v": i2v_id,
                "image": by_class["LoadImage"][0]}

    def customize(self, template: dict, *, prompt: str, negative_prompt: str,
                  seed: int, width: int, height: int, length: int,
                  image_name: str) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["positive"]]["inputs"]["text"] = prompt
        wf[found["negative"]]["inputs"]["text"] = negative_prompt
        wf[found["sampler"]]["inputs"]["seed"] = seed
        wf[found["i2v"]]["inputs"].update(
            {"width": width, "height": height, "length": length})
        wf[found["image"]]["inputs"]["image"] = image_name
        return wf

    def generate(self, request: VideoRequest) -> VideoResult:
        if request.input_image is not None and not os.path.isfile(request.input_image):
            raise ProviderError(self.provider_id,
                                f"Input image not found: {request.input_image}.")
        width, height = self.adjust_dimensions(request.width, request.height)
        length = self.frames_for_duration(request.duration)
        seed = (request.seed if request.seed is not None
                else random.randint(0, 2**31 - 1))
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            stored = self.client.upload_image(str(request.input_image))
            wf = self.customize(template, prompt=request.prompt,
                                negative_prompt=request.negative_prompt, seed=seed,
                                width=width, height=height, length=length,
                                image_name=stored)
            dest = self.client.run(wf, ("gifs", "images", "video"), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=width, height=height)
