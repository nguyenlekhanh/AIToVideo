"""Unit tests: dimensions, registry lookup, unknown models (no server)."""
from __future__ import annotations

import os
import unittest

from providers import dims
from providers.errors import ProviderError, UnknownModelError
from providers.registry import list_models, lookup

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DimsTest(unittest.TestCase):
    def test_landscape_720(self):
        self.assertEqual(dims.resolve_dimensions("16:9", 720), (1312, 736))

    def test_portrait_720(self):
        self.assertEqual(dims.resolve_dimensions("9:16", 720), (736, 1312))

    def test_square(self):
        self.assertEqual(dims.resolve_dimensions("1:1", 420), (416, 416))

    def test_snap(self):
        self.assertEqual(dims.snap(720, 32), 736)
        self.assertEqual(dims.snap(746, 32), 736)
        self.assertEqual(dims.snap(512, 16), 512)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            dims.parse_aspect("wide")
        with self.assertRaises(ValueError):
            dims.parse_aspect("16:0")
        with self.assertRaises(ValueError):
            dims.parse_resolution("hd")
        with self.assertRaises(ValueError):
            dims.resolve_dimensions("16:9", 100)


class RegistryTest(unittest.TestCase):
    def test_lookup_image_flux(self):
        entry = lookup(AI_VIDEO_DIR, "image", "flux")
        self.assertEqual(entry["provider"], "flux")
        self.assertTrue(entry["workflow_path"].endswith("flux.json"))

    def test_lookup_video_ltx(self):
        entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        self.assertEqual(entry["provider"], "ltx")
        self.assertTrue(entry["workflow_path"].endswith("ltx2_5_i2v.json"))

    def test_lookup_krea_resolves_absolute_paths(self):
        # Regression: e2e from another CWD failed because workflow_ref stayed
        # relative. Registry must hand out absolute, existing paths.
        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        for key in ("workflow_path", "workflow_ref_path"):
            with self.subTest(key=key):
                self.assertTrue(os.path.isabs(entry[key]), key)
                self.assertTrue(os.path.isfile(entry[key]), key)

    def test_lookup_audio(self):
        entry = lookup(AI_VIDEO_DIR, "audio", "edge_tts")
        self.assertEqual(entry["provider"], "edge_tts")

    def test_unknown_model(self):
        with self.assertRaises(UnknownModelError) as ctx:
            lookup(AI_VIDEO_DIR, "video", "sora")
        self.assertIn("Available", str(ctx.exception))

    def test_unknown_kind_model(self):
        with self.assertRaises(UnknownModelError):
            lookup(AI_VIDEO_DIR, "image", "krea2")

    def test_list_models(self):
        self.assertEqual(set(list_models(AI_VIDEO_DIR, "image")), {"sd15", "sdxl", "flux", "krea"})
        self.assertEqual(set(list_models(AI_VIDEO_DIR, "video")), {"ltx", "wan"})

    def test_provider_error_format(self):
        err = ProviderError("ltx", "boom", workflow="ltx2_5_i2v.json", detail="trace...")
        text = str(err)
        self.assertIn("ltx", text)
        self.assertIn("ltx2_5_i2v.json", text)
        self.assertIn("boom", text)
        self.assertIn("trace...", text)


if __name__ == "__main__":
    unittest.main()
