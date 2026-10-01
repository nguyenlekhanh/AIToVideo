"""Image provider interface. main.py only knows these types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int | None
    output_path: Path
    reference_image_path: Path | None = None


@dataclass
class ImageResult:
    path: Path
    width: int
    height: int


class ImageProvider(ABC):
    provider_id: str = "image"

    @abstractmethod
    def generate(self, request: ImageRequest) -> ImageResult:
        """Generate a keyframe image. Returns path + final dimensions used."""

    def adjust_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Validate/adjust requested dims for this model. Default: identity."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}.")
        return width, height
