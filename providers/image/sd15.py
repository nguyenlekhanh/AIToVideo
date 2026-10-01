"""SD 1.5 keyframe provider (backward-compatible default).

Uses the original verified SD1.5 text-to-image workflow.
"""
from __future__ import annotations

from .comfyui import SdCheckpointImageProvider


class Sd15ImageProvider(SdCheckpointImageProvider):
    provider_id = "sd15"
