"""Model-independent dimension handling.

The application resolves --aspect/--resolution to a target (width, height).
Each provider then validates/adjusts for its model's constraints and reports
the final dimensions it actually used.
"""
from __future__ import annotations

import math


def parse_aspect(aspect: str) -> tuple[int, int]:
    try:
        w, h = aspect.strip().split(":")
        aw, ah = int(w), int(h)
    except ValueError:
        raise ValueError(f"Invalid --aspect {aspect!r}: expected 'W:H', e.g. '16:9'.")
    if aw <= 0 or ah <= 0:
        raise ValueError(f"Invalid --aspect {aspect!r}: parts must be positive.")
    return aw, ah


def parse_resolution(resolution: int | str) -> int:
    try:
        n = int(resolution)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid --resolution {resolution!r}: expected e.g. 420 or 720.")
    if n < 144 or n > 2048:
        raise ValueError(f"Invalid --resolution {resolution!r}: expected 144..2048.")
    return n


def snap(value: int, multiple: int) -> int:
    """Round to nearest multiple (ties up). Multiple must be >= 1."""
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return int(math.floor(value / multiple + 0.5)) * multiple


# ResolutionSelector aspect labels (ComfyUI core node) and their ratios.
ASPECT_LABELS = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def select_aspect_label(width: int, height: int) -> tuple[str, int, int]:
    """Closest ResolutionSelector aspect label for the given dimensions."""
    ratio = width / height
    label = min(ASPECT_LABELS,
                key=lambda k: abs(math.log((ASPECT_LABELS[k][0] / ASPECT_LABELS[k][1])
                                           / ratio)))
    aw, ah = ASPECT_LABELS[label]
    return label, aw, ah


def resolve_dimensions(aspect: str, resolution: int | str, multiple: int = 32,
                       ) -> tuple[int, int]:
    """Resolve (aspect, resolution) to (width, height).

    Resolution N is the frame height for landscape aspects and the frame
    width for portrait aspects; results are snapped to `multiple`.
    """
    aw, ah = parse_aspect(aspect)
    n = parse_resolution(resolution)
    if aw >= ah:  # landscape or square
        height = snap(n, multiple)
        width = snap(round(height * aw / ah), multiple)
    else:  # portrait
        width = snap(n, multiple)
        height = snap(round(width * ah / aw), multiple)
    if width < multiple or height < multiple:
        raise ValueError(f"Resolved dimensions too small: {width}x{height}.")
    return width, height
