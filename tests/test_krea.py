"""Unit + live tests for the Krea provider.

Live tests need ComfyUI (skipped otherwise): TestKreaLiveValidate only
queues (job cancelled immediately); TestKreaLiveGenerate renders one real
9:16/720p image through the Krea-2 Turbo workflow.
"""
from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path

from providers.comfy import ComfyClient
from providers.errors import ProviderError
from providers.image.base import ImageRequest
from providers.image.krea import KreaImageProvider
from providers.registry import create_provider, lookup

from tests import COMFY_URL, require_comfy

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIVE_PROMPT = (
    "A cinematic lone astronaut standing in an ancient abandoned Chinese city "
    "buried beneath the red sands of Mars, dramatic sunset, dust storm, "
    "detailed architecture, cinematic science fiction"
)


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "image", "krea")
    settings = dict(entry, **overrides)
    return KreaImageProvider(name="krea", settings=settings,
                             client=ComfyClient(COMFY_URL))


class TestKreaUnit(unittest.TestCase):
    def test_registry_lookup(self):
        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        self.assertEqual(entry["provider"], "krea")
        self.assertTrue(entry["workflow_path"].endswith("krea2_turbo_t2i.json"))

    def test_create_via_registry(self):
        provider = create_provider("image", AI_VIDEO_DIR, "krea",
                                   client=ComfyClient(COMFY_URL))
        self.assertIsInstance(provider, KreaImageProvider)

    def test_workflow_loads_with_required_nodes(self):
        wf = make_provider().load_workflow()
        self.assertEqual(len(wf), 22)
        classes = {n.get("class_type") for n in wf.values()}
        for required in ("UNETLoader", "CLIPLoader", "VAELoader", "KSampler",
                         "EmptyLatentImage", "VAEDecode", "SaveImage",
                         "ConditioningZeroOut", "ResolutionSelector",
                         "PrimitiveStringMultiline", "TextGenerate",
                         "LoraLoaderModelOnly"):
            self.assertIn(required, classes)

    def test_prompt_substitution_scope(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(template, prompt="P", seed=7,
                                   aspect_label="16:9 (Widescreen)",
                                   megapixels=0.5)
        allowed = {"PrimitiveStringMultiline", "KSampler", "ResolutionSelector"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        changed = [nid for nid in before if before[nid] != after[nid]]
        self.assertEqual(len(changed), 3)  # user prompt, KSampler seed, selector
        # the substituted prompt node feeds the enhance/encode chain
        self.assertEqual(
            after[provider._find_nodes(template)["prompt"]]["inputs"]["value"], "P")

    def test_negative_empty_ok_nonempty_rejected(self):
        # Empty negative is accepted (checked again in the live test, which
        # passes negative_prompt="" through generate()).
        provider = make_provider()
        with self.assertRaises(ProviderError):
            bad = ImageRequest(prompt="p",
                               negative_prompt="blurry, low quality, distorted, "
                                               "watermark, text, bad anatomy",
                               width=736, height=1312, seed=1,
                               output_path=Path("x.png"))
            provider.generate(bad)

    def test_reference_image_missing_file(self):
        provider = make_provider()
        req = ImageRequest(prompt="p", negative_prompt="", width=736,
                           height=1312, seed=1, output_path=Path("x.png"),
                           reference_image_path=Path("does-not-exist.png"))
        with self.assertRaises(ProviderError) as ctx:
            provider.generate(req)
        self.assertIn("Reference image not found", str(ctx.exception))

    def test_reference_without_registered_workflow(self):
        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        settings = dict(entry)
        settings.pop("workflow_ref", None)
        settings.pop("workflow_ref_path", None)
        provider = KreaImageProvider(
            name="krea", settings=settings, client=ComfyClient(COMFY_URL))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            ref = f.name
        try:
            req = ImageRequest(prompt="p", negative_prompt="", width=736,
                               height=1312, seed=1, output_path=Path("x.png"),
                               reference_image_path=Path(ref))
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(req)
            self.assertIn("workflow_ref", str(ctx.exception))
        finally:
            os.unlink(ref)

    def test_reference_customization_scope(self):
        from providers.image.krea import REQUIRED_REF_NODE_CLASSES
        provider = make_provider()
        template = provider.load_workflow(provider.workflow_ref_path,
                                          REQUIRED_REF_NODE_CLASSES)
        before = copy.deepcopy(template)
        after = provider.customize_ref(template, prompt="P", seed=7,
                                       image_name="ref.png", denoise=0.65)
        allowed = {"PrimitiveStringMultiline", "KSampler", "LoadImage"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        found = provider._find_ref_nodes(template)
        self.assertEqual(after[found["prompt"]]["inputs"]["value"], "P")
        self.assertEqual(after[found["sampler"]]["inputs"]["denoise"], 0.65)
        self.assertEqual(after[found["image"]]["inputs"]["image"], "ref.png")
        latent = after[found["sampler"]]["inputs"]["latent_image"]
        self.assertEqual(after[str(latent[0])]["class_type"], "VAEEncode")

    def test_ref_finder_rejects_t2i_workflow(self):
        provider = make_provider()
        with self.assertRaises(ProviderError):
            provider._find_ref_nodes(provider.load_workflow())

    def test_invalid_denoise_rejected(self):
        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        with self.assertRaises(ValueError):
            KreaImageProvider(name="krea",
                              settings=dict(entry, denoise=1.5),
                              client=ComfyClient(COMFY_URL))

    def test_generate_reports_probed_dims(self):
        # T2I generate() must report the file's actual dims, not the
        # selector estimate: stub the client to "render" a 100x200 PNG.
        import struct

        from providers.comfy import ComfyClient

        def png_bytes(w, h):
            ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
            return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
                    + ihdr + struct.pack(">I", 0))

        class StubClient(ComfyClient):
            def run(self, workflow, output_keys, dest_path):
                with open(dest_path, "wb") as f:
                    f.write(png_bytes(100, 200))
                return dest_path

        provider = make_provider()
        provider.client = StubClient(COMFY_URL)
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "stub.png")
            result = provider.generate(ImageRequest(
                prompt="p", negative_prompt="", width=736, height=1312,
                seed=1, output_path=Path(dest)))
            self.assertEqual((result.width, result.height), (100, 200))

    def test_probe_image_size_png(self):
        import struct
        ihdr = struct.pack(">IIBBBBB", 752, 1336, 8, 2, 0, 0, 0)
        png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr
               + struct.pack(">I", 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png)
            path = f.name
        try:
            self.assertEqual(KreaImageProvider.probe_image_size(path), (752, 1336))
        finally:
            os.unlink(path)

    def test_invalid_workflow_detected(self):
        provider = make_provider()
        template = provider.load_workflow()
        broken = {nid: node for nid, node in template.items()
                  if node.get("class_type") != "TextGenerate"}
        with self.assertRaises(ProviderError) as ctx:
            provider.customize(broken, prompt="P", seed=1,
                               aspect_label="9:16 (Portrait Widescreen)",
                               megapixels=1.0)
        self.assertIn("TextGenerate", str(ctx.exception))

    def test_missing_workflow_file(self):
        provider = make_provider(
            workflow_path=os.path.join(AI_VIDEO_DIR, "nope.json"))
        with self.assertRaises(ProviderError):
            provider.load_workflow()

    def test_adjust_dimensions(self):
        provider = make_provider()
        self.assertEqual(provider.adjust_dimensions(736, 1312), (736, 1312))
        self.assertEqual(provider.adjust_dimensions(730, 1300), (728, 1304))
        with self.assertRaises(ValueError):
            provider.adjust_dimensions(0, 512)

    def test_selector_settings_portrait(self):
        provider = make_provider()
        label, mp, final = provider.selector_settings(736, 1312)
        self.assertEqual(label, "9:16 (Portrait Widescreen)")
        self.assertEqual(mp, 1.0)
        self.assertEqual(final[0] % 8, 0)
        self.assertEqual(final[1] % 8, 0)


class TestKreaLiveValidate(unittest.TestCase):
    def test_validate_graph(self):
        require_comfy(self)
        provider = make_provider()
        wf = provider.customize(provider.load_workflow(), prompt="a cat",
                                seed=1,
                                aspect_label="9:16 (Portrait Widescreen)",
                                megapixels=0.5)
        client = ComfyClient(COMFY_URL)
        prompt_id = client.queue(wf)
        self.addCleanup(client.cancel, prompt_id)
        self.assertTrue(prompt_id)


class TestKreaLiveGenerate(unittest.TestCase):
    def test_generate_real_image(self):
        require_comfy(self)
        provider = make_provider()
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "krea_live.png")
            result = provider.generate(ImageRequest(
                prompt=LIVE_PROMPT, negative_prompt="", width=736,
                height=1312, seed=1234, output_path=Path(dest)))
            self.assertTrue(os.path.isfile(dest))
            with open(dest, "rb") as f:
                header = f.read(8)
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")
            size = os.path.getsize(dest)
            self.assertGreater(size, 50 * 1024)
            print(f"\nLIVE KREA: provider=krea workflow=krea2_turbo_t2i.json "
                  f"requested=736x1312 actual={result.width}x{result.height} "
                  f"seed=1234 path={dest} size={size} status=success")


if __name__ == "__main__":
    unittest.main()
