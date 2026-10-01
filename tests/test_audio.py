"""Unit tests: audio provider (creation + empty-text error; synthesis is live)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from providers.audio.base import AudioRequest
from providers.audio.edge_tts import EdgeTtsProvider
from providers.errors import ProviderError
from providers.registry import create_provider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AudioTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("audio", AI_VIDEO_DIR, "edge_tts")
        self.assertIsInstance(provider, EdgeTtsProvider)

    def test_empty_text_rejected(self):
        provider = create_provider("audio", AI_VIDEO_DIR, "edge_tts")
        with self.assertRaises(ProviderError):
            provider.generate(AudioRequest(text="  ", output_path=Path("x.mp3")))


if __name__ == "__main__":
    unittest.main()
