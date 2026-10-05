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
        allowed = {"PrimitiveStringMultiline", "KSampler", "ResolutionSelector",
                   "PrimitiveBoolean"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        changed = [nid for nid in before if before[nid] != after[nid]]
        self.assertEqual(len(changed), 4)  # user prompt, KSampler seed, selector, enhance switch
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
        allowed = {"PrimitiveStringMultiline", "KSampler", "LoadImage",
                   "PrimitiveBoolean"}
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


def trace_prompt_terminals(workflow):
    """Follow every CLIPTextEncode text input through ComfySwitchNodes
    (honoring each PrimitiveBoolean value) to its terminal node.

    Returns {encode_id: (terminal_class, visited_node_ids)}.
    """
    out = {}
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "CLIPTextEncode":
            continue
        src = node.get("inputs", {}).get("text")
        visited = []
        cur = str(src[0]) if isinstance(src, list) else None
        while cur is not None and cur not in visited:
            visited.append(cur)
            cur_node = workflow.get(cur, {})
            cur_class = cur_node.get("class_type")
            if cur_class == "PreviewAny":
                # transparent passthrough: follow its source input
                src_in = cur_node.get("inputs", {}).get("source")
                cur = str(src_in[0]) if isinstance(src_in, list) else None
                continue
            if cur_class != "ComfySwitchNode":
                break
            sw = cur_node.get("inputs", {}).get("switch")
            boolean = workflow.get(str(sw[0]), {}) if isinstance(sw, list) else {}
            value = boolean.get("inputs", {}).get("value", True)
            nxt = cur_node.get("inputs", {}).get(
                "on_true" if value else "on_false")
            cur = str(nxt[0]) if isinstance(nxt, list) else None
        terminal = workflow.get(visited[-1], {}).get("class_type") \
            if visited else None
        out[nid] = (terminal, visited)
    return out


class EnhancerBypassTest(unittest.TestCase):
    @staticmethod
    def _enhancer_switch(workflow):
        """(switch_id, bool_id) of the switch routing TextGenerate output."""
        by_class = {}
        for nid, node in workflow.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)
        enhancers = by_class.get("TextGenerate", [])
        assert len(enhancers) == 1
        for nid in by_class.get("ComfySwitchNode", []):
            on_true = workflow[nid].get("inputs", {}).get("on_true")
            if isinstance(on_true, list) and str(on_true[0]) == enhancers[0]:
                return nid, str(workflow[nid]["inputs"]["switch"][0])
        raise AssertionError("no enhancer switch found")

    def test_customize_disables_enhancer(self):
        provider = make_provider()
        template = provider.load_workflow()
        switch_id, bool_id = self._enhancer_switch(template)
        self.assertIs(template[switch_id]["inputs"] and
                      template[bool_id]["inputs"]["value"], True)
        wf = provider.customize(template, prompt="P", seed=7,
                                aspect_label="16:9 (Widescreen)",
                                megapixels=0.5)
        self.assertIs(wf[bool_id]["inputs"]["value"], False)
        # template object untouched by the in-memory flip
        self.assertIs(template[bool_id]["inputs"]["value"], True)

    def test_conditioning_path_avoids_text_generate(self):
        provider = make_provider()
        wf = provider.customize(provider.load_workflow(), prompt="P", seed=7,
                                aspect_label="16:9 (Widescreen)",
                                megapixels=0.5)
        terminals = trace_prompt_terminals(wf)
        self.assertTrue(terminals)
        for encode_id, (terminal, visited) in terminals.items():
            visited_classes = [wf[n].get("class_type") for n in visited]
            self.assertNotIn(
                "TextGenerate", visited_classes,
                f"conditioning path for {encode_id} still passes the enhancer")
            self.assertEqual(
                terminal, "PrimitiveStringMultiline",
                f"conditioning path for {encode_id} must end at the raw prompt")

    def test_ref_path_avoids_text_generate(self):
        from providers.image.krea import REQUIRED_REF_NODE_CLASSES
        provider = make_provider()
        wf = provider.customize_ref(
            provider.load_workflow(provider.workflow_ref_path,
                                   REQUIRED_REF_NODE_CLASSES),
            prompt="P", seed=7, image_name="ref.png", denoise=0.65)
        terminals = trace_prompt_terminals(wf)
        self.assertTrue(terminals)
        for encode_id, (terminal, visited) in terminals.items():
            self.assertNotIn(
                "TextGenerate",
                [wf[n].get("class_type") for n in visited])
            self.assertEqual(terminal, "PrimitiveStringMultiline")

    def test_template_file_unchanged_on_disk(self):
        import hashlib
        provider = make_provider()
        path = provider.workflow_path
        with open(path, "rb") as f:
            before = hashlib.sha256(f.read()).hexdigest()
        template = provider.load_workflow()
        provider.customize(template, prompt="P", seed=7,
                           aspect_label="16:9 (Widescreen)", megapixels=0.5)
        with open(path, "rb") as f:
            after = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(before, after)
        # and the template default is still enhance-ON (change is runtime-only)
        raw = provider.load_workflow()
        by_class = {}
        for nid, node in raw.items():
            by_class.setdefault(node.get("class_type", ""), []).append(nid)
        true_switches = [
            nid for nid in by_class.get("ComfySwitchNode", [])
            if raw[nid]["inputs"].get("on_true", [None])[0]
            in by_class.get("TextGenerate", [])
            and raw[raw[nid]["inputs"]["switch"][0]]["inputs"]["value"] is True]
        self.assertTrue(true_switches)


if __name__ == "__main__":
    unittest.main()
