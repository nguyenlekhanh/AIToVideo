"""Wan Animate2 distilled motion-control provider.

Graph (transcribed from workflows_template/video_wan_animate2_distilled.json
by export_wan_animate2_template.py): reference LoadImage -> cleanGpuUsed ->
WanAnimate2ToVideo subgraph (pose LoadVideo drives motion) -> frames ->
CreateVideo (+pose audio/fps) -> SaveVideo.

Fixed ~5s output (template length 81); the provider declares
fixed_duration so the pipeline validates existence, not duration.
One motion clip = one output; the SAME reference image is reused for
every job (uploaded once, cached by path).
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

from ..comfy import ComfyClient, ComfyError
from ..errors import ProviderError
from .base import VideoProvider, VideoRequest, VideoResult

MOTION_EXTENSIONS = (".mp4", ".mov", ".webm")
OUTPUT_KEYS = ("gifs", "video", "images")


def _natural_key(path: str) -> list:
    import re
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


class WanAnimate2Provider(VideoProvider):
    provider_id = "wan_animate2"

    # Pipeline contract flags (read by main.py; LTX-style providers omit them).
    fixed_duration = True      # template length is fixed (~5s); don't enforce storyboard durations
    uses_motion_clips = True   # each scene consumes one driving motion clip
    supports_pinned_reference = True  # one reference image reused for every job

    def __init__(self, name: str = "wan_animate2",
                 settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))
        self._upload_cache: dict[str, str] = {}

    @staticmethod
    def discover_motion_clips(directory: str) -> list[str]:
        """Natural-sorted motion clips. Clear errors, never silent."""
        if not os.path.isdir(directory):
            raise ProviderError("wan_animate2",
                                f"Motion directory not found: {directory}.")
        found = [os.path.join(directory, n) for n in os.listdir(directory)
                 if os.path.isfile(os.path.join(directory, n))
                 and os.path.splitext(n)[1].lower() in MOTION_EXTENSIONS]
        if not found:
            raise ProviderError(
                "wan_animate2",
                f"No motion clips (.mp4/.mov/.webm) found in {directory}.")
        found.sort(key=_natural_key)
        return found

    def load_workflow(self) -> dict:
        if not self.workflow_path or not os.path.isfile(self.workflow_path):
            raise ProviderError(self.provider_id, "Workflow file not found.",
                                workflow=self.workflow_path or "(unset)")
        with open(self.workflow_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        by_class: dict[str, list[str]] = {}
        for node_id, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(node_id)
        loads = by_class.get("LoadImage", [])
        if len(loads) != 1:
            raise ProviderError("wan_animate2",
                                f"Expected 1 LoadImage node, found {loads}.")
        videos = by_class.get("LoadVideo", [])
        if len(videos) != 1:
            raise ProviderError("wan_animate2",
                                f"Expected 1 LoadVideo node, found {videos}.")
        prompts = by_class.get("PrimitiveStringMultiline", [])
        if len(prompts) != 2:
            raise ProviderError("wan_animate2",
                                f"Expected 2 prompt nodes, found {prompts}.")
        saves = by_class.get("SaveVideo", [])
        if len(saves) != 1:
            raise ProviderError("wan_animate2",
                                f"Expected 1 SaveVideo node, found {saves}.")
        seeds = by_class.get("SamplerCustom", [])
        if len(seeds) != 1:
            raise ProviderError("wan_animate2",
                                f"Expected 1 SamplerCustom node, found {seeds}.")
        # Character prompt = the PrimitiveStringMultiline feeding the
        # character (slot 4) branch; pose prompt feeds slot 6. Resolve via
        # links: inner 1003 text <- 605, inner 1222 text <- 604.
        char = pose = None
        for node_id in prompts:
            for target, _, _ in _outgoing(workflow, node_id):
                if target == "1003":
                    char = node_id
                elif target == "1222":
                    pose = node_id
        if char is None or pose is None:
            raise ProviderError("wan_animate2",
                                "Character/pose prompt wiring not found.")
        return {"image": loads[0], "video": videos[0], "char": char,
                "pose": pose, "save": saves[0], "sampler": seeds[0]}

    def customize(self, template: dict, *, reference_name: str,
                  motion_name: str, character_prompt: str, seed: int,
                  prefix: str) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["image"]]["inputs"]["image"] = reference_name
        wf[found["video"]]["inputs"]["file"] = motion_name
        wf[found["char"]]["inputs"]["value"] = character_prompt
        wf[found["sampler"]]["inputs"]["noise_seed"] = seed
        wf[found["save"]]["inputs"]["filename_prefix"] = prefix
        return wf

    def _upload_once(self, local_path: str, kind: str) -> str:
        cached = self._upload_cache.get(local_path)
        if cached is not None:
            return cached
        if kind == "video":
            stored = self.client.upload_video(local_path)
        else:
            stored = self.client.upload_image(local_path)
        self._upload_cache[local_path] = stored
        return stored

    def generate(self, request: VideoRequest) -> VideoResult:
        if request.input_image is not None and not os.path.isfile(request.input_image):
            raise ProviderError(self.provider_id,
                                f"Reference image not found: {request.input_image}.")
        motion = request.motion_video
        if motion is None or not os.path.isfile(str(motion)):
            raise ProviderError(self.provider_id,
                                f"Motion clip missing for this scene: {motion}. "
                                f"wan_animate2 needs --motion-dir.")
        seed = (request.seed if request.seed is not None
                else random.randint(0, 2**31 - 1))
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            ref_name = self._upload_once(str(request.input_image), "image")
            motion_name = self._upload_once(str(motion), "video")
            stem = os.path.splitext(os.path.basename(str(motion)))[0]
            wf = self.customize(template, reference_name=ref_name,
                                motion_name=motion_name,
                                character_prompt=request.prompt,
                                seed=seed, prefix=f"wan_animate2/{stem}")
            dest = self.client.run(wf, OUTPUT_KEYS, str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Video generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return VideoResult(path=Path(dest), width=request.width,
                           height=request.height)


def _outgoing(workflow: dict, node_id: str) -> list[tuple[str, int, int]]:
    """(target_id, target_slot, origin_slot) for links leaving node_id."""
    results = []
    for target_id, target in workflow.items():
        for _name, value in (target.get("inputs") or {}).items():
            if isinstance(value, list) and len(value) == 2 \
                    and str(value[0]) == str(node_id):
                results.append((str(target_id), value[1], 0))
    return results
