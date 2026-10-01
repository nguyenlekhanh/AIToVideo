"""Unit tests: LTX provider creation, workflow loading, substitution scope, dims."""
from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path

from providers.comfy import ComfyClient, ComfyError
from providers.errors import ProviderError
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.ltx import LtxVideoProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "video", "ltx")
    settings = dict(entry, **overrides)
    return LtxVideoProvider(name="ltx", settings=settings,
                            client=ComfyClient("http://127.0.0.1:8188"))


class LtxCreationTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("video", AI_VIDEO_DIR, "ltx",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, LtxVideoProvider)
        self.assertEqual(provider.provider_id, "ltx")

    def test_workflow_loads_with_required_nodes(self):
        wf = make_provider().load_workflow()
        self.assertEqual(len(wf), 50)
        classes = {n.get("class_type") for n in wf.values()}
        for required in ("EmptyLTXVLatentVideo", "LTXVConcatAVLatent",
                         "LTXVSeparateAVLatent", "SamplerCustomAdvanced",
                         "LTXVImgToVideoInplace", "LTXVPreprocess", "LoadImage"):
            self.assertIn(required, classes)

    def test_missing_workflow_raises_provider_error(self):
        provider = make_provider(workflow_path=os.path.join(AI_VIDEO_DIR, "nope.json"))
        with self.assertRaises(ProviderError) as ctx:
            provider.load_workflow()
        self.assertIn("ltx", str(ctx.exception))


class LtxSubstitutionTest(unittest.TestCase):
    def test_only_allowed_inputs_change(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(
            template, video_prompt="P", negative_prompt="N", seed=7,
            duration=4, image_name="x.png",
            aspect_label="3:2 (Photo)", megapixels=0.4)
        allowed = {"PrimitiveStringMultiline", "CLIPTextEncode", "RandomNoise",
                   "PrimitiveInt", "LoadImage", "ResolutionSelector"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        # every substituted node is present exactly once (except 2 seeds)
        changed = [nid for nid in before if before[nid] != after[nid]]
        self.assertEqual(len(changed), 6)  # prompt, neg, 2x seed, duration, image, resolution

    def test_resolution_is_applied(self):
        provider = make_provider()
        after = provider.customize(
            provider.load_workflow(), video_prompt="P", negative_prompt="N",
            seed=7, duration=4, image_name="x.png",
            aspect_label="9:16 (Portrait Widescreen)", megapixels=1.0)
        sel = [n for n in after.values() if n.get("class_type") == "ResolutionSelector"]
        self.assertEqual(len(sel), 1)
        self.assertEqual(sel[0]["inputs"]["aspect_ratio"], "9:16 (Portrait Widescreen)")
        self.assertEqual(sel[0]["inputs"]["megapixels"], 1.0)


class LtxDimsTest(unittest.TestCase):
    def test_snap_to_32(self):
        provider = make_provider()
        self.assertEqual(provider.adjust_dimensions(1312, 736), (1312, 736))
        self.assertEqual(provider.adjust_dimensions(1300, 700), (1312, 704))
        with self.assertRaises(ValueError):
            provider.adjust_dimensions(0, 512)

    def test_selector_settings_known_good(self):
        provider = make_provider()
        label, mp, final = provider.selector_settings(768, 512)
        self.assertEqual(label, "3:2 (Photo)")
        self.assertEqual(mp, 0.4)
        self.assertEqual(final, (768, 512))  # matches verified render

    def test_selector_settings_portrait(self):
        provider = make_provider()
        label, mp, final = provider.selector_settings(736, 1312)
        self.assertEqual(label, "9:16 (Portrait Widescreen)")
        self.assertEqual(mp, 1.0)
        self.assertEqual(final, (768, 1344))  # matches verified render

    def test_bad_duration_rejected(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="n", input_image=None,
                           width=768, height=512, duration=99, seed=1,
                           output_path=Path("x.mp4"))
        with self.assertRaises(ValueError):
            provider.generate(req)

    def test_missing_image_raises_provider_error(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="n",
                           input_image=Path("does-not-exist.png"),
                           width=768, height=512, duration=5, seed=1,
                           output_path=Path("x.mp4"))
        with self.assertRaises(ProviderError) as ctx:
            provider.generate(req)
        self.assertIn("[provider:ltx]", str(ctx.exception))

    def test_comfy_error_wrapped_with_context(self):
        class BoomClient(ComfyClient):
            def upload_image(self, image_path: str) -> str:
                raise ComfyError("connection reset")
        entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        provider = LtxVideoProvider(name="ltx", settings=dict(entry),
                                      client=BoomClient("http://127.0.0.1:8188"))
        req = VideoRequest(prompt="p", negative_prompt="n",
                           input_image=Path(__file__),
                           width=768, height=512, duration=5, seed=1,
                           output_path=Path("x.mp4"))
        with self.assertRaises(ProviderError) as ctx:
            provider.generate(req)
        text = str(ctx.exception)
        self.assertIn("[provider:ltx]", text)
        self.assertIn("ltx2_5_i2v.json", text)
        self.assertIn("connection reset", text)


if __name__ == "__main__":
    unittest.main()
