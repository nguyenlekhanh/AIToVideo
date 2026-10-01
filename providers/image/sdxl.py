"""SDXL keyframe provider (same checkpoint graph shape, SDXL weights)."""
from __future__ import annotations

from .comfyui import SdCheckpointImageProvider


class SdxlImageProvider(SdCheckpointImageProvider):
    provider_id = "sdxl"
