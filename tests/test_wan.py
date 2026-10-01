"""Unit + validation tests for the Wan provider (validation needs ComfyUI)."""
from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

from providers.comfy import ComfyClient
from providers.registry import create_provider
from providers.video.base import VideoRequest
from providers.video.wan import WanVideoProvider

from tests import COMFY_URL, require_comfy

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider():
    return create_provider("video", AI_VIDEO_DIR, "wan",
                           client=ComfyClient(COMFY_URL))


class WanTest(unittest.TestCase):
    def test_create_via_registry(self):
        self.assertIsInstance(make_provider(), WanVideoProvider)

    def test_frames_math(self):
        self.assertEqual(WanVideoProvider.frames_for_duration(5), 81)
        self.assertEqual(WanVideoProvider.frames_for_duration(3), 49)
        self.assertEqual((WanVideoProvider.frames_for_duration(5) - 1) % 4, 0)

    def test_adjust_dimensions(self):
        provider = make_provider()
        self.assertEqual(provider.adjust_dimensions(832, 480), (832, 480))
        self.assertEqual(provider.adjust_dimensions(830, 479), (832, 480))

    def test_substitution_scope(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(template, prompt="P", negative_prompt="N",
                                   seed=3, width=832, height=480, length=81,
                                   image_name="x.png")
        allowed = {"CLIPTextEncode", "KSampler", "WanImageToVideo", "LoadImage"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed)

    def test_validate_graph(self):
        require_comfy(self)
        provider = make_provider()
        wf = provider.customize(provider.load_workflow(), prompt="a cat",
                                negative_prompt="blurry", seed=1, width=832,
                                height=480, length=5, image_name="ai_video_test.png")
        client = ComfyClient(COMFY_URL)
        prompt_id = client.queue(wf)
        self.addCleanup(client.cancel, prompt_id)
        self.assertTrue(prompt_id)


if __name__ == "__main__":
    unittest.main()
