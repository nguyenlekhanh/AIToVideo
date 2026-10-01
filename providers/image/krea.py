"""Krea-2 Turbo text-to-image provider.

Uses the template export workflows/image/krea2_turbo_t2i.json EXACTLY:
UNETLoader (krea2_turbo_fp8) + Qwen3VL CLIPLoader + Qwen VAE + optional
style LoRA switch + prompt-enhance LLM switch + KSampler + VAEDecode.

Substitutable inputs ONLY: user prompt, KSampler seed, ResolutionSelector
(requested dimensions). The graph has NO negative-prompt path (the positive
conditioning is zeroed for the negative side), and NO image input (T2I only).
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
from .base import ImageProvider, ImageRequest, ImageResult

DIM_MULTIPLE = 8  # template ResolutionSelector multiple

REQUIRED_NODE_CLASSES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "KSampler",
    "EmptyLatentImage",
    "VAEDecode",
    "SaveImage",
    "ConditioningZeroOut",
    "ResolutionSelector",
    "PrimitiveStringMultiline",
)

# Character-reference (img2img) graph: same loaders/conditioning/enhance/VAE
# as the T2I graph, but LoadImage -> VAEEncode feeds KSampler instead of
# EmptyLatentImage (dims come from the reference image).
REQUIRED_REF_NODE_CLASSES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "KSampler",
    "LoadImage",
    "VAEEncode",
    "VAEDecode",
    "SaveImage",
    "ConditioningZeroOut",
    "PrimitiveStringMultiline",
)


class KreaImageProvider(ImageProvider):
    provider_id = "krea"

    def __init__(self, name: str = "krea", settings: dict | None = None,
                 client: ComfyClient | None = None):
        self.name = name
        self.settings = settings or {}
        self.workflow_path = self.settings.get("workflow_path", "")
        # workflow_ref_path is absolute (built by the registry); fall back to
        # the raw workflow_ref value for directly-constructed providers.
        self.workflow_ref_path = self.settings.get("workflow_ref_path", "") or \
            self.settings.get("workflow_ref", "")
        self.denoise = float(self.settings.get("denoise", 0.65))
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError(f"Krea denoise must be in (0, 1]: {self.denoise}.")
        self.client = client or ComfyClient(
            self.settings.get("comfy_url", "http://127.0.0.1:8188"))

    # -- dimensions: snap to the selector multiple, then predict the
    # selector's height-first output (same formula as the LTX provider). --
    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return snap(width, DIM_MULTIPLE), snap(height, DIM_MULTIPLE)

    def selector_settings(self, width: int, height: int) -> tuple[str, float, int]:
        width, height = self.adjust_dimensions(width, height)
        label, aw, ah = select_aspect_label(width, height)
        megapixels = max(0.1, round(width * height / 1_000_000, 1))
        unit = math.sqrt(megapixels * 1_000_000 / (aw * ah))
        final_h = snap(round(ah * unit), DIM_MULTIPLE)
        final_w = snap(round(final_h * aw / ah), DIM_MULTIPLE)
        return label, megapixels, (final_w, final_h)

    # -- workflow handling (structure-aware, id-independent) --
    def load_workflow(self, path: str | None = None,
                      required: tuple[str, ...] = REQUIRED_NODE_CLASSES) -> dict:
        path = path or self.workflow_path
        if not path or not os.path.isfile(path):
            raise ProviderError(self.provider_id, "Workflow file not found.",
                                workflow=path or "(unset)")
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
        wf.pop("_meta", None)
        present = {n.get("class_type") for n in wf.values()}
        missing = [c for c in required if c not in present]
        if missing:
            raise ProviderError(
                self.provider_id,
                f"Workflow is not the expected Krea-2 graph; missing: {missing}.",
                workflow=os.path.basename(path))
        return wf

    @staticmethod
    def _find_nodes(workflow: dict) -> dict:
        """Locate prompt / seed / resolution nodes by graph structure.

        The user-prompt node is the PrimitiveStringMultiline feeding string_b
        of the StringConcatenate that feeds TextGenerate.prompt (the other
        one is the fixed system prompt). Raises ProviderError if the graph
        does not expose a supported input (fail loud, never mis-generate).
        """
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)

        samplers = by_class.get("KSampler", [])
        if len(samplers) != 1:
            raise ProviderError("krea", f"Expected 1 KSampler node, found {samplers}.")
        selectors = by_class.get("ResolutionSelector", [])
        if len(selectors) != 1:
            raise ProviderError("krea", f"Expected 1 ResolutionSelector node, found {selectors}.")
        enhancers = by_class.get("TextGenerate", [])
        if len(enhancers) != 1:
            raise ProviderError("krea", f"Expected 1 TextGenerate node, found {enhancers}.")
        prompt_src = workflow[enhancers[0]]["inputs"].get("prompt")
        if not isinstance(prompt_src, list):
            raise ProviderError("krea", "TextGenerate.prompt input not found.")
        concat = workflow.get(str(prompt_src[0]))
        if concat is None or concat.get("class_type") != "StringConcatenate":
            raise ProviderError(
                "krea",
                "Workflow image_krea2_turbo_t2i.json does not expose a supported "
                "positive prompt input (expected StringConcatenate into TextGenerate).")
        user_src = concat["inputs"].get("string_b")
        if not isinstance(user_src, list):
            raise ProviderError("krea", "Prompt StringConcatenate.string_b input not found.")
        user_node = workflow.get(str(user_src[0]))
        if user_node is None or user_node.get("class_type") != "PrimitiveStringMultiline":
            raise ProviderError(
                "krea",
                "Workflow image_krea2_turbo_t2i.json does not expose a supported "
                "positive prompt input (expected PrimitiveStringMultiline).")
        return {"prompt": str(user_src[0]), "sampler": samplers[0],
                "resolution": selectors[0]}

    @staticmethod
    def _find_ref_nodes(workflow: dict) -> dict:
        """Locate prompt / seed / reference-input nodes in the character
        (img2img) graph. Same prompt-chain lookup as T2I; dims come from the
        reference image, so no ResolutionSelector is expected."""
        by_class: dict[str, list[str]] = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)

        samplers = by_class.get("KSampler", [])
        if len(samplers) != 1:
            raise ProviderError("krea", f"Expected 1 KSampler node, found {samplers}.")
        loads = by_class.get("LoadImage", [])
        if len(loads) != 1:
            raise ProviderError("krea", f"Expected 1 LoadImage node, found {loads}.")
        encodes = by_class.get("VAEEncode", [])
        if len(encodes) != 1:
            raise ProviderError("krea", f"Expected 1 VAEEncode node, found {encodes}.")
        sampler_latent = workflow[samplers[0]]["inputs"].get("latent_image")
        if not (isinstance(sampler_latent, list)
                and workflow.get(str(sampler_latent[0]), {}).get("class_type") == "VAEEncode"):
            raise ProviderError(
                "krea",
                "Character workflow KSampler.latent_image is not fed by VAEEncode; "
                "refusing to guess the reference path.")
        enhancers = by_class.get("TextGenerate", [])
        if len(enhancers) != 1:
            raise ProviderError("krea", f"Expected 1 TextGenerate node, found {enhancers}.")
        prompt_src = workflow[enhancers[0]]["inputs"].get("prompt")
        if not isinstance(prompt_src, list):
            raise ProviderError("krea", "TextGenerate.prompt input not found.")
        concat = workflow.get(str(prompt_src[0]))
        if concat is None or concat.get("class_type") != "StringConcatenate":
            raise ProviderError(
                "krea",
                "Workflow krea2_character_ref.json does not expose a supported "
                "positive prompt input (expected StringConcatenate into TextGenerate).")
        user_src = concat["inputs"].get("string_b")
        user_node = workflow.get(str(user_src[0])) if isinstance(user_src, list) else None
        if user_node is None or user_node.get("class_type") != "PrimitiveStringMultiline":
            raise ProviderError(
                "krea",
                "Workflow krea2_character_ref.json does not expose a supported "
                "positive prompt input (expected PrimitiveStringMultiline).")
        return {"prompt": str(user_src[0]), "sampler": samplers[0],
                "image": loads[0]}

    def customize(self, template: dict, *, prompt: str, seed: int,
                  aspect_label: str, megapixels: float) -> dict:
        wf = copy.deepcopy(template)
        found = self._find_nodes(wf)
        wf[found["prompt"]]["inputs"]["value"] = prompt
        wf[found["sampler"]]["inputs"]["seed"] = seed
        wf[found["resolution"]]["inputs"] = {
            "aspect_ratio": aspect_label, "megapixels": megapixels, "multiple": 8}
        return wf

    def customize_ref(self, template: dict, *, prompt: str, seed: int,
                      image_name: str, denoise: float) -> dict:
        """Substitute the character-reference graph inputs only."""
        wf = copy.deepcopy(template)
        found = self._find_ref_nodes(wf)
        wf[found["prompt"]]["inputs"]["value"] = prompt
        wf[found["sampler"]]["inputs"]["seed"] = seed
        wf[found["sampler"]]["inputs"]["denoise"] = denoise
        wf[found["image"]]["inputs"]["image"] = image_name
        return wf

    @staticmethod
    def probe_image_size(path: str) -> tuple[int, int]:
        """Read PNG/JPEG dimensions from file headers (stdlib only)."""
        with open(path, "rb") as f:
            head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            import struct
            w, h = struct.unpack(">II", head[16:24])
            return w, h
        if head[:2] == b"\xff\xd8":
            import struct
            with open(path, "rb") as f:
                data = f.read()
            i = 2
            while i < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7) or marker == 0x01:
                    i += 2
                else:
                    i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
        raise ProviderError("krea", f"Cannot read image dimensions (PNG/JPEG only): {path}.")

    # -- ImageProvider interface --
    def generate(self, request: ImageRequest) -> ImageResult:
        if (request.negative_prompt or "").strip():
            raise ProviderError(
                self.provider_id,
                "The Krea-2 Turbo workflows have no negative prompt input "
                "(negative side is ConditioningZeroOut); refusing non-empty negative_prompt.")
        if request.reference_image_path is not None:
            return self._generate_with_reference(request)
        width, height = self.adjust_dimensions(request.width, request.height)
        aspect_label, megapixels, (exp_w, exp_h) = self.selector_settings(width, height)
        seed = (request.seed if request.seed is not None
                else random.randint(0, 2**31 - 1))
        wf_name = os.path.basename(self.workflow_path)
        try:
            template = self.load_workflow()
            wf = self.customize(template, prompt=request.prompt, seed=seed,
                                aspect_label=aspect_label, megapixels=megapixels)
            dest = self.client.run(wf, ("images",), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Image generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        # Report the ACTUAL output dims (probed), not the selector estimate:
        # the selector's exact rounding is only verified empirically.
        try:
            final_w, final_h = self.probe_image_size(dest)
        except ProviderError:
            final_w, final_h = exp_w, exp_h
        return ImageResult(path=Path(dest), width=final_w, height=final_h)

    def _generate_with_reference(self, request: ImageRequest) -> ImageResult:
        """Character path: same Krea graph as img2img on the reference image."""
        ref = str(request.reference_image_path)
        if not os.path.isfile(ref):
            raise ProviderError(self.provider_id, f"Reference image not found: {ref}.")
        if not self.workflow_ref_path or not os.path.isfile(self.workflow_ref_path):
            raise ProviderError(
                self.provider_id,
                "No character-reference workflow registered (models.json "
                "'workflow_ref'); cannot use reference_image_path.")
        ref_w, ref_h = self.probe_image_size(ref)
        seed = (request.seed if request.seed is not None
                else random.randint(0, 2**31 - 1))
        wf_name = os.path.basename(self.workflow_ref_path)
        try:
            template = self.load_workflow(self.workflow_ref_path, REQUIRED_REF_NODE_CLASSES)
            stored = self.client.upload_image(ref)
            wf = self.customize_ref(template, prompt=request.prompt, seed=seed,
                                    image_name=stored, denoise=self.denoise)
            dest = self.client.run(wf, ("images",), str(request.output_path))
        except ComfyError as exc:
            raise ProviderError(self.provider_id, "Character image generation failed.",
                                workflow=wf_name, detail=str(exc)) from exc
        return ImageResult(path=Path(dest), width=ref_w, height=ref_h)
