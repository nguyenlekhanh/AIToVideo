"""Video provider interface. main.py only knows these types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoRequest:
    prompt: str
    negative_prompt: str
    input_image: Path | None
    width: int
    height: int
    duration: float  # seconds; provider converts to frames for its model
    seed: int | None
    output_path: Path
    motion_video: Path | None = None  # driving clip (wan_animate2 only)


@dataclass
class VideoResult:
    path: Path
    width: int
    height: int


class VideoProvider(ABC):
    provider_id: str = "video"

    @abstractmethod
    def generate(self, request: VideoRequest) -> VideoResult:
        """Generate a video scene. Returns path + final dimensions used."""

    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Validate/adjust requested dims for this model. Default: identity."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return width, height
