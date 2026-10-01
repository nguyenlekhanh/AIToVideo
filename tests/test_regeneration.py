"""Unit tests: bounded single-scene regeneration (no Ollama, no network)."""
from __future__ import annotations

import unittest

import main
import storyboard as sb


def meta_with_fact():
    return {
        "mode": "web", "sources": [],
        "facts": [{"id": "fact_001", "claim": "Atlas can jump high.",
                   "source_ids": ["src_001"], "published_at": None,
                   "freshness": "unknown", "confidence": 0.7}],
        "conflicts": [],
        "source_records": [{"id": "src_001", "title": "T",
                            "url": "https://example.com/x",
                            "publisher": "Example", "published_at": None,
                            "retrieved_at": "2026-01-01T00:00:00+00:00",
                            "source_type": "news"}],
    }


def scene(narration, fid="fact_001"):
    return {"id": 1, "duration": 5, "image_prompt": "p", "video_prompt": "v",
            "narration": narration, "research_fact_ids": [fid],
            "source_ids": ["src_001"]}


class RegenerationLoopTest(unittest.TestCase):
    def test_pass_first_try_no_rewrites(self):
        calls = []

        def rewrite_fn(scene, facts, reason):
            calls.append((scene, facts, reason))
            return scene

        scenes = [scene("Atlas can jump high.")]
        out, rewrites, warnings = main.ensure_grounded_scenes(
            scenes, meta_with_fact(), rewrite_fn)
        self.assertEqual(calls, [])
        self.assertEqual(rewrites, 0)
        self.assertEqual(out[0]["narration"], "Atlas can jump high.")

    def test_rewrite_then_pass(self):
        calls = []

        def rewrite_fn(scn, facts, reason):
            calls.append(reason)
            fixed = dict(scn)
            fixed["narration"] = "Atlas can jump high."
            return fixed

        scenes = [scene("Atlas can jump 50 meters high over tall skyscrapers.")]
        out, rewrites, warnings = main.ensure_grounded_scenes(
            scenes, meta_with_fact(), rewrite_fn)
        self.assertEqual(rewrites, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("50", calls[0])
        self.assertEqual(out[0]["narration"], "Atlas can jump high.")
        self.assertEqual(out[0]["research_fact_ids"], ["fact_001"])

    def test_persistent_failure_raises_bounded(self):
        calls = []

        def rewrite_fn(scn, facts, reason):
            calls.append(reason)
            return dict(scn)  # never improves

        scenes = [scene("Atlas can jump 50 meters high over tall skyscrapers.")]
        with self.assertRaises(ValueError) as ctx:
            main.ensure_grounded_scenes(scenes, meta_with_fact(), rewrite_fn,
                                        max_rounds=2)
        self.assertEqual(len(calls), 2)
        self.assertIn("Scene #1", str(ctx.exception))

    def test_no_research_noop(self):
        calls = []

        def rewrite_fn(scn, facts, reason):
            calls.append(reason)
            return scn

        out, rewrites, warnings = main.ensure_grounded_scenes(
            [scene("Anything at all.")], None, rewrite_fn)
        self.assertEqual((calls, rewrites, warnings), ([], 0, []))

    def test_partial_warns_without_rewrite(self):
        calls = []

        def rewrite_fn(scn, facts, reason):
            calls.append(reason)
            return scn

        scenes = [scene("Atlas can jump and fly over tall buildings.")]
        out, rewrites, warnings = main.ensure_grounded_scenes(
            scenes, meta_with_fact(), rewrite_fn)
        self.assertEqual(rewrites, 0)
        self.assertTrue(warnings)
        self.assertEqual(out, scenes)

    def test_warnings_not_duplicated_across_rounds(self):
        calls = []

        def rewrite_fn(scn, facts, reason):
            calls.append(reason)
            fixed = dict(scn)
            fixed["narration"] = "Atlas can jump high."
            return fixed

        scenes = [scene("Atlas can jump 50 meters high over tall skyscrapers."),
                  scene("Atlas can jump and fly over tall buildings.")]
        _, rewrites, warnings = main.ensure_grounded_scenes(
            scenes, meta_with_fact(), rewrite_fn)
        self.assertEqual(rewrites, 1)
        self.assertEqual(len(warnings), len(set(warnings)))


class RewriteSceneTest(unittest.TestCase):
    def test_message_shape_and_preservation(self):
        import ollama as ol_mod
        captured = {}
        real = ol_mod._post

        def fake_post(url, payload, timeout):
            captured.update(payload)
            return {"message": {"content": '{"narration": "Atlas can jump high."}'},
                    "model": "m"}

        ol_mod._post = fake_post
        try:
            updated = ol_mod.rewrite_scene_narration(
                scene("Something else entirely.", fid="fact_001"),
                [{"id": "fact_001", "claim": "Atlas can jump high."}],
                "unsupported terms: else, entirely",
                model="m", base_url="http://x", timeout=5)
        finally:
            ol_mod._post = real
        self.assertEqual(updated["narration"], "Atlas can jump high.")
        self.assertEqual(updated["research_fact_ids"], ["fact_001"])
        self.assertEqual(updated["duration"], 5)
        user_text = captured["messages"][1]["content"]
        self.assertIn("fact_001", user_text)
        self.assertIn("Atlas can jump high.", user_text)
        self.assertIn("unsupported terms", user_text)

    def test_invalid_json_raises(self):
        import ollama as ol_mod
        real = ol_mod._post
        ol_mod._post = lambda *a, **k: {"message": {"content": "nope"}, "model": "m"}
        try:
            with self.assertRaises(RuntimeError):
                ol_mod.rewrite_scene_narration(
                    scene("x"), [{"id": "fact_001", "claim": "y"}],
                    "reason", model="m", base_url="http://x", timeout=5)
        finally:
            ol_mod._post = real

    def test_empty_narration_raises(self):
        import ollama as ol_mod
        real = ol_mod._post
        ol_mod._post = lambda *a, **k: {"message": {"content": '{"narration": "  "}'},
                                        "model": "m"}
        try:
            with self.assertRaises(RuntimeError):
                ol_mod.rewrite_scene_narration(
                    scene("x"), [{"id": "fact_001", "claim": "y"}],
                    "reason", model="m", base_url="http://x", timeout=5)
        finally:
            ol_mod._post = real


if __name__ == "__main__":
    unittest.main()
