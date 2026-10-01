"""Flux keyframe provider (same single-file checkpoint graph shape).

Uses a Flux schnell checkpoint with few-step sampling. Sampling defaults
(steps/cfg/sampler) come from the registry entry.
"""
from __future__ import annotations

from .comfyui import SdCheckpointImageProvider


class FluxImageProvider(SdCheckpointImageProvider):
    provider_id = "flux"
