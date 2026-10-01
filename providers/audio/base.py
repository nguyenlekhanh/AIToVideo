"""Audio provider interface. main.py only knows these types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioRequest:
    text: str
    output_path: Path
    voice: str = ""
    rate: str = "+0%"
    pitch: str = "+0Hz"


@dataclass
class AudioResult:
    path: Path


class AudioProvider(ABC):
    provider_id: str = "audio"

    @abstractmethod
    def generate(self, request: AudioRequest) -> AudioResult:
        """Synthesize narration audio."""
