"""Regression tests: storyboard metadata must never leak into image prompts.

No GPU/network: prompt construction and guard behavior only (plus one
mocked-transport Krea test asserting the exact payload).
"""
from __future__ import annotations

import json
import os
import struct
import unittest
from pathlib import Path

from subject import (
    SubjectProfile,
    assert_no_schema_leak,
    compose_scene_prompt,
)

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUBJECT = {
    "type": "person",
    "identity": "determined individual",
    "features": "determined eyes, faint scar",
    "clothing_or_equipment": "weathered jacket",
    "consistency": "unchanging appearance",
}
SCENE_TEXT = "A determined person walking toward the sunrise."
FORBIDDEN = ("Recurring subject:", "type: person", "identity:",
             "features:", "clothing_or_equipment:", "consistency:")


class LeakContractTest(unittest.TestCase):
    def test_spec_example_clean(self):
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        assert profile is not None
        prompt = compose_scene_prompt(profile, SCENE_TEXT, with_reference=True)
        self.assertIn(SCENE_TEXT, prompt)
        for leaked in FORBIDDEN:
            with self.subTest(leaked=leaked):
                self.assertNotIn(leaked, prompt)
                self.assertNotIn(leaked.lower(), prompt.lower())

    def test_no_raw_storyboard_json(self):
        scene = {"id": 3, "duration": 5, "image_prompt": SCENE_TEXT,
                 "video_prompt": "pan", "narration": "Dawn breaks.",
                 "research_fact_ids": ["fact_001"], "source_ids": ["src_001"],
                 "grounding": "researched"}
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        prompt = compose_scene_prompt(profile, scene["image_prompt"],
                                      with_reference=True)
        self.assertNotIn(json.dumps(scene), prompt)
        for token in ("{", '"id"', '"duration"', "fact_001", "src_001",
                      "researched"):
            with self.subTest(token=token):
                self.assertNotIn(token, prompt)

    def test_no_research_metadata(self):
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        prompt = compose_scene_prompt(profile, SCENE_TEXT, with_reference=True)
        for token in ("research_fact_ids", "source_ids", "grounding",
                      "fact_001", "src_001"):
            with self.subTest(token=token):
                self.assertNotIn(token, prompt)

    def test_no_scene_ids(self):
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        prompt = compose_scene_prompt(profile, SCENE_TEXT, with_reference=True)
        self.assertNotIn("scene_003", prompt)
        self.assertNotIn('"id"', prompt)

    def test_natural_words_preserved(self):
        text = ("The scene features exotic features and various types "
                "of lanterns glowing at dusk.")
        self.assertEqual(assert_no_schema_leak(text), text)
        self.assertEqual(compose_scene_prompt(None, text), text)

    def test_guard_rejects_labels(self):
        for bad in ("Scene:\ntype: person\nfeatures: determined eyes",
                    "RECURRING SUBJECT: TYPE: PERSON",
                    "A photo. clothing_or_equipment: jacket",
                    "Shot list.\nreference image: match this",
                    '{"image_prompt": "a cat"}'):
            with self.subTest(bad=bad[:30]):
                with self.assertRaises(ValueError):
                    assert_no_schema_leak(bad)


class KreaPayloadTest(unittest.TestCase):
    def test_exact_krea_prompt(self):
        from providers.comfy import ComfyClient
        from providers.image.krea import KreaImageProvider
        from providers.registry import lookup

        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        provider = KreaImageProvider(
            name="krea", settings=dict(entry),
            client=ComfyClient("http://127.0.0.1:8188"))
        template = provider.load_workflow()
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        composed = compose_scene_prompt(profile, SCENE_TEXT, with_reference=True)
        customized = provider.customize(
            template, prompt=composed, seed=7,
            aspect_label="9:16 (Portrait Widescreen)", megapixels=1.0)
        found = provider._find_nodes(customized)
        node_text = customized[found["prompt"]]["inputs"]["value"]
        self.assertEqual(node_text, composed)
        assert_no_schema_leak(node_text)

    def test_generate_end_to_end_prompt(self):
        import tempfile
        from providers.comfy import ComfyClient
        from providers.image.base import ImageRequest
        from providers.image.krea import KreaImageProvider
        from providers.registry import lookup

        captured = {}

        class StubClient(ComfyClient):
            def run(self, workflow, output_keys, dest_path):
                node = workflow[KreaImageProvider._find_nodes(workflow)["prompt"]]
                captured["text"] = node["inputs"]["value"]
                ihdr = struct.pack(">IIBBBBB", 752, 1336, 8, 2, 0, 0, 0)
                png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13)
                       + b"IHDR" + ihdr + struct.pack(">I", 0))
                Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
                Path(dest_path).write_bytes(png)
                return dest_path

        entry = lookup(AI_VIDEO_DIR, "image", "krea")
        provider = KreaImageProvider(name="krea", settings=dict(entry),
                                    client=StubClient("http://127.0.0.1:8188"))
        import tempfile
        profile = SubjectProfile.from_dict(dict(SUBJECT))
        composed = compose_scene_prompt(profile, SCENE_TEXT, with_reference=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = provider.generate(ImageRequest(
                prompt=composed, negative_prompt="", width=736, height=1312,
                seed=7, output_path=Path(tmp) / "out.png"))
        self.assertEqual(captured["text"], composed)
        self.assertEqual((result.width, result.height), (752, 1336))
        assert_no_schema_leak(captured["text"])


class RealStoryboardFixtureTest(unittest.TestCase):
    def test_all_scenes_of_real_storyboards(self):
        checked = 0
        for name in ("hoian_girl", "subject_girl", "mars_city"):
            path = os.path.join(AI_VIDEO_DIR, "projects", name, "storyboard.json")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            profile = SubjectProfile.from_dict(raw.get("subject"))
            for scene in raw.get("scenes", []):
                prompt = compose_scene_prompt(
                    profile, scene["image_prompt"], with_reference=True)
                self.assertIn(scene["image_prompt"], prompt)
                checked += 1
        self.assertGreater(checked, 0, "expected at least one storyboard fixture")


if __name__ == "__main__":
    unittest.main()
