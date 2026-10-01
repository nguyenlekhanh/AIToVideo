"""ComfyUI checkpoint image providers.

Covers single-file checkpoint models with the standard graph shape:
CheckpointLoaderSimple -> CLIPTextEncode x2 -> EmptyLatentImage ->
KSampler -> VAEDecode -> SaveImage.
(sdxl.py / flux.py set model-specific defaults; the graph shape is shared.)
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
from .base import ImageProvider, ImageRequest, ImageResult


class SdCheckpointImageProvider(ImageProvider):
    provider_id = "sd_checkpoint"
    dim_multiple = 8

    def __init__(self, name: str = "sd", settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        self.steps = int(self.settings.get("steps", 20))
        self.cfg = float(self.settings.get("cfg", 7.0))
        self.sampler = str(self.settings.get("sampler", "euler"))
        self.scheduler = str(self.settings.get("scheduler", "normal"))
        self.dim_multiple = int(self.settings.get("dim_multiple", self.dim_multiple))
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))

    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return snap(width, self.dim_multiple), snap(height, self.dim_multiple)

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
        for cls, want in (("CheckpointLoaderSimple", 1), ("EmptyLatentImage", 1),
                          ("KSampler", 1), ("VAEDecode", 1), ("SaveImage", 1)):
            if len(by_class.get(cls, [])) != want:
                raise ProviderError("image",
                                    f"Expected {want}x {cls}, found {by_class.get(cls, [])}.")
        samplers = by_class["KSampler"]
        sampler_inputs = workflow[samplers[0]]["inputs"]
        try:
            pos_id = str(sampler_inputs["positive"][0])
            neg_id = str(sampler_inputs["negative"][0])
        except (KeyError, TypeError, IndexError):
            raise ProviderError("image", "KSampler positive/negative links not found.")
        if workflow[pos_id].get("class_type") != "CLIPTextEncode":
            raise ProviderError("image", "KSampler positive input is not CLIPTextEncode.")
        if workflow[neg_id].get("class_type") != "CLIPTextEncode":
            raise ProviderError("image", "KSampler negative input is not CLIPTextEncode.")
        return {"positive": pos_id, "negative": neg_id,
                "latent": by_class["EmptyLatentImage"][0],
                "sampler": samplers[0]}

    def customize(self, template: dict, *, prompt: str, negative_prompt: str,
                  seed: int, width: int, height: int) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["positive"]]["inputs"]["text"] = prompt
        wf[found["negative"]]["inputs"]["text"] = negative_prompt
        wf[found["latent"]]["inputs"]["width"] = width
        wf[found["latent"]]["inputs"]["height"] = height
        wf[found["sampler"]]["inputs"].update(
            {"seed": seed, "steps": self.steps, "cfg": self.cfg,
             "sampler_name": self.sampler, "scheduler": self.scheduler})
        return wf

    def generate(self, request: ImageRequest) -> ImageResult:
        width, height = self.adjust_dimensions(request.width, request.height)
        seed = (request.seed if request.seed is not None
                else random.randint(0, 2**31 - 1))
        wf_name = os.path.basename(self.workflow_path)
        try:
            wf = self.customize(self.load_workflow(), prompt=request.prompt,
                                negative_prompt=request.negative_prompt,
                                seed=seed, width=width, height=height)
            dest = self.client.run(wf, ("images",), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Image generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return ImageResult(path=Path(dest), width=width, height=height)
