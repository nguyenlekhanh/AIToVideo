"""Unit + validation tests for image providers.

Validation tests submit each model's customized graph to ComfyUI /prompt:
a prompt_id means the graph is valid (job is cancelled immediately, nothing
renders). Needs ComfyUI running; skipped otherwise.
"""
from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path

from providers.comfy import ComfyClient
from providers.image.base import ImageRequest
from providers.image.comfyui import SdCheckpointImageProvider
from providers.registry import create_provider, lookup

from tests import COMFY_URL, require_comfy

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider(model: str):
    return create_provider("image", AI_VIDEO_DIR, model,
                           client=ComfyClient(COMFY_URL))


class ImageCreationTest(unittest.TestCase):
    def test_create_all_models(self):
        for model in ("sd15", "sdxl", "flux"):
            with self.subTest(model=model):
                provider = create_provider("image", AI_VIDEO_DIR, model,
                                           client=ComfyClient(COMFY_URL))
                self.assertIsInstance(provider, SdCheckpointImageProvider)
                self.assertTrue(os.path.isfile(provider.workflow_path))

    def test_substitution_scope(self):
        provider = create_provider("image", AI_VIDEO_DIR, "flux",
                                   client=ComfyClient(COMFY_URL))
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(template, prompt="P", negative_prompt="N",
                                   seed=3, width=1312, height=736)
        allowed = {"CLIPTextEncode", "EmptyLatentImage", "KSampler"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed)
        self.assertEqual(after["2"]["inputs"]["text"], "P")

    def test_adjust_dimensions(self):
        sd15 = create_provider("image", AI_VIDEO_DIR, "sd15",
                               client=ComfyClient(COMFY_URL))
        self.assertEqual(sd15.adjust_dimensions(1300, 700), (1304, 704))
        flux = create_provider("image", AI_VIDEO_DIR, "flux",
                               client=ComfyClient(COMFY_URL))
        self.assertEqual(flux.adjust_dimensions(1312, 736), (1312, 736))
        with self.assertRaises(ValueError):
            sd15.adjust_dimensions(-1, 512)


class ImageValidationTest(unittest.TestCase):
    def validate_model(self, model: str):
        require_comfy(self)
        provider = create_provider("image", AI_VIDEO_DIR, model,
                                   client=ComfyClient(COMFY_URL))
        wf = provider.customize(provider.load_workflow(), prompt="a cat",
                                negative_prompt="blurry", seed=1,
                                width=512, height=512)
        client = ComfyClient(COMFY_URL)
        prompt_id = client.queue(wf)
        self.addCleanup(client.cancel, prompt_id)
        self.assertTrue(prompt_id)

    def test_validate_sd15(self):
        self.validate_model("sd15")

    def test_validate_sdxl(self):
        self.validate_model("sdxl")

    def test_validate_flux(self):
        self.validate_model("flux")


if __name__ == "__main__":
    unittest.main()
