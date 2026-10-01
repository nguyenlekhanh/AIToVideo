"""Edge TTS narration provider (one file per scene)."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ..errors import ProviderError
from .base import AudioProvider, AudioRequest, AudioResult


class EdgeTtsProvider(AudioProvider):
    provider_id = "edge_tts"

    def __init__(self, name: str = "edge_tts", settings: dict | None = None,
                 **kwargs):
        self.name = name
        self.settings = settings or {}

    def generate(self, request: AudioRequest) -> AudioResult:
        text = (request.text or "").strip()
        if not text:
            raise ProviderError(self.provider_id, "Cannot synthesize empty narration.")
        try:
            import edge_tts
        except ImportError as exc:
            raise ProviderError(self.provider_id,
                                "The 'edge-tts' package is not installed.") from exc

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                text, voice=request.voice or "en-US-AriaNeural",
                rate=request.rate, pitch=request.pitch)
            await communicate.save(str(request.output_path))

        os.makedirs(os.path.dirname(os.path.abspath(request.output_path)), exist_ok=True)
        try:
            asyncio.run(_run())
        except Exception as exc:
            raise ProviderError(self.provider_id, "Edge TTS synthesis failed.",
                                detail=str(exc)[:1000]) from exc
        return AudioResult(path=Path(request.output_path))
