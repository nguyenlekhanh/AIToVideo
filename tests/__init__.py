"""Test package. Run with:  python -m unittest discover -s tests   (from ai_video/)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMFY_URL = os.environ.get("AI_VIDEO_COMFY_URL", "http://127.0.0.1:8188")


def require_comfy(testcase: unittest.TestCase):
    """Skip the test unless ComfyUI is reachable."""
    from providers.comfy import ComfyClient, ComfyError
    try:
        ComfyClient(COMFY_URL, timeout=10).check_reachable(timeout=5)
    except ComfyError as exc:
        testcase.skipTest(f"ComfyUI not reachable: {exc}")
