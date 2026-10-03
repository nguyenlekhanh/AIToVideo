"""Unit tests: Mode C script-to-storyboard (long text -> Qwen -> storyboard).

No GPU/network: the Ollama transport (ollama._post) and model-availability
checks are mocked. Text-only requests are asserted (no "images" payload).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import ollama as ol
import script as sc
import storyboard as sb
from subject import assert_no_schema_leak

CONDENSED = {
    "title": "The Shepherd",
    "summary": "A shepherd searches for one lost sheep.",
    "key_events": ["Sheep wanders off", "Shepherd searches at night",
                   "Shepherd carries it home"],
    "characters": ["The shepherd"],
    "locations": ["Hills", "Village"],
    "video_script": [
        {"narration": "One sheep wandered from the flock.",
         "visual": "A sheep alone on dark hills.",
         "source_reference": "opening"},
        {"narration": "The shepherd searched through the night.",
         "visual": "A lantern moving across hills at night.",
         "source_reference": "middle"},
        {"narration": "He carried it home rejoicing.",
         "visual": "Shepherd carrying a sheep toward warm lights.",
         "source_reference": "ending"},
    ],
}


def storyboard_response(n=3, duration=7, subject=None, narration="Voice."):
    return {"message": {"content": json.dumps({
        "subject": subject or {"type": "none", "identity": "",
                               "features": "", "clothing_or_equipment": "",
                               "consistency": ""},
        "scenes": [{
            "id": i + 1, "duration": duration,
            "image_prompt": f"A cinematic scene of moment {i + 1}.",
            "video_prompt": f"The camera drifts slowly in moment {i + 1}.",
            "narration": narration} for i in range(n)],
    })}}


def condense_response(**overrides):
    body = dict(CONDENSED)
    body.update(overrides)
    return {"message": {"content": json.dumps(body)}}


def write_source(text="The shepherd sought the lost sheep."):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                      encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name


class ScriptCliTest(unittest.TestCase):
    def test_parses_script(self):
        args = main.parse_args(["--project", "b", "--script", "bible.txt"])
        self.assertEqual(args.script, "bible.txt")

    def test_subject_default_auto(self):
        args = main.parse_args(["--project", "b", "--script", "bible.txt"])
        self.assertEqual(args.subject, "auto")

    def test_parses_subject(self):
        args = main.parse_args(["--project", "b", "--script", "bible.txt",
                                "--subject", "god"])
        self.assertEqual(args.subject, "god")

    def test_mode_c_default_model(self):
        args = main.parse_args(["--project", "b", "--script", "bible.txt"])
        self.assertEqual(
            getattr(args, "analysis_model", None) or "qwen3:8b", "qwen3:8b")

    def test_prompt_optional_with_script(self):
        args = main.parse_args(["--project", "b", "--script", "s.txt"])
        self.assertIsNone(args.prompt)
        self.assertIsNone(main.validate_cli_combination(args))

    def test_script_keyframes_mutually_exclusive(self):
        args = main.parse_args(["--project", "b", "--script", "s.txt",
                                "--keyframes", "kf"])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_script_prompt_conflict(self):
        args = main.parse_args(["idea", "--project", "b", "--script", "s.txt"])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_script_resume_conflict(self):
        args = main.parse_args(["--project", "b", "--resume",
                                "--script", "s.txt"])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_script_requires_project(self):
        args = main.parse_args(["--script", "s.txt"])
        self.assertIsNotNone(main.validate_cli_combination(args))


class SourceHandlingTest(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            sc.load_source_text("/nonexistent/source.txt")

    def test_empty_file(self):
        path = write_source("   \n  ")
        try:
            with self.assertRaisesRegex(ValueError, "empty"):
                sc.load_source_text(path)
        finally:
            os.unlink(path)

    def test_normalize(self):
        self.assertEqual(sc.normalize_text("  Hello   world.\n\n\nNext."),
                         "Hello world.\n\nNext.")

    def test_short_text_single_chunk(self):
        chunks = sc.chunk_text("First sentence. Second sentence.")
        self.assertEqual(len(chunks), 1)

    def test_long_text_splits_on_sentences(self):
        text = " ".join(f"Sentence number {i} here." for i in range(200))
        chunks = sc.chunk_text(text, max_chars=400)
        self.assertGreater(len(chunks), 1)
        rejoined = " ".join(chunks)
        for i in (0, 100, 199):
            self.assertIn(f"Sentence number {i} here.", rejoined)

    def test_oversized_sentence_hard_split(self):
        chunks = sc.chunk_text("x" * 5000, max_chars=4000)
        self.assertEqual(len(chunks), 2)


class CondenseTest(unittest.TestCase):
    def test_short_source_single_request(self):
        with patch.object(ol, "_post",
                           return_value=condense_response()) as post:
            out = sc.condense_source("Short text.", model="m")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(out["video_script"]), 3)
        self.assertEqual(out["key_events"][0], "Sheep wanders off")

    def test_long_source_chunked(self):
        text = " ".join(f"Verse {i} tells of the journey onward."
                        for i in range(400))
        self.assertGreater(len(text), sc.SINGLE_LIMIT_CHARS)
        extract = {"message": {"content": json.dumps({
            "key_events": ["An event"], "characters": ["Someone"],
            "locations": ["Somewhere"], "notes": "A note."})}}
        calls = []

        def fake_post(url, payload, timeout):
            calls.append(payload)
            content = payload["messages"][0]["content"]
            if "ONE chunk" in content:
                return extract
            return condense_response()

        with patch.object(ol, "_post", side_effect=fake_post):
            out = sc.condense_source(text, model="m")
        # chunk extracts + one final condensation
        self.assertGreater(len(calls), 2)
        self.assertTrue(out["video_script"])
        for payload in calls:
            for message in payload["messages"]:
                self.assertNotIn("images", message)

    def test_malformed_condensation_retried_then_fails(self):
        bad = {"message": {"content": "not json"}}
        with patch.object(ol, "_post", return_value=bad) as post:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                sc.condense_source("Text.", model="m")
        self.assertEqual(post.call_count, 3)

    def test_empty_video_script_rejected(self):
        with patch.object(ol, "_post",
                           return_value=condense_response(video_script=[])):
            with self.assertRaisesRegex(RuntimeError, "no video_script"):
                sc.condense_source("Text.", model="m")

    def test_invalid_content_retried_then_succeeds(self):
        bad = {"message": {"content": json.dumps({
            "title": "T", "summary": "S", "key_events": ["e"],
            "characters": [], "locations": [],
            "video_script": [{"narration": " ", "visual": "v",
                              "source_reference": "r"}]})}}
        calls = {"n": 0}

        def flaky(url, payload, timeout):
            calls["n"] += 1
            return bad if calls["n"] < 2 else condense_response()

        with patch.object(ol, "_post", side_effect=flaky):
            out = sc.condense_source("Text.", model="m")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(out["video_script"]), 3)

    def test_persisted_condensed_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "condensed_script.json")
            sc.save_condensed_script(CONDENSED, path)
            loaded = json.load(open(path, encoding="utf-8"))
        self.assertEqual(loaded["title"], "The Shepherd")
        self.assertEqual(len(loaded["video_script"]), 3)


class StoryboardPlanningTest(unittest.TestCase):
    def _plan(self, **kwargs):
        kwargs.setdefault("model", "m")
        with patch.object(ol, "_post",
                           return_value=storyboard_response(
                               n=kwargs.pop("n", 3),
                               duration=kwargs.pop("duration", 7))), \
             patch.object(ol, "check_model_available", return_value=None):
            return sc.plan_storyboard(dict(CONDENSED), **kwargs)

    def test_scene_count_and_shape(self):
        result = self._plan(num_scenes=3)
        self.assertEqual([s["id"] for s in result["scenes"]], [1, 2, 3])
        self.assertIsNone(result["research"])
        for scene in result["scenes"]:
            self.assertEqual(scene["duration"], 7)
            self.assertTrue(scene["narration"].strip())
        validated = sb.validate_storyboard(result)
        self.assertEqual(len(validated), 3)

    def test_duration_clamped_to_range(self):
        result = self._plan(num_scenes=3, duration=99)
        for scene in result["scenes"]:
            self.assertGreaterEqual(scene["duration"], 5)
            self.assertLessEqual(scene["duration"], 10)

    def test_low_duration_clamped(self):
        result = self._plan(num_scenes=3, duration=1)
        for scene in result["scenes"]:
            self.assertGreaterEqual(scene["duration"], 5)

    def test_god_preset_propagated(self):
        result = self._plan(num_scenes=3,
                            subject=dict(sc.SUBJECT_PRESETS["god"]))
        self.assertEqual(result["subject"]["identity"], "God")
        self.assertEqual(result["subject"]["type"], "character")

    def test_auto_subject_none_for_empty(self):
        result = self._plan(num_scenes=3, subject=None)
        self.assertEqual(result["subject"]["type"], "none")

    def test_wrong_scene_count_retried_then_fails(self):
        with patch.object(ol, "_post",
                           return_value=storyboard_response(n=2)) as post:
            with self.assertRaisesRegex(RuntimeError, "Storyboard planning"):
                sc.plan_storyboard(dict(CONDENSED), model="m", num_scenes=3)
        self.assertEqual(post.call_count, 3)

    def test_empty_narration_retried_then_fails(self):
        with patch.object(ol, "_post",
                           return_value=storyboard_response(narration=" ")):
            with self.assertRaisesRegex(RuntimeError, "Storyboard planning"):
                sc.plan_storyboard(dict(CONDENSED), model="m", num_scenes=3)

    def test_prompts_pass_leak_guard(self):
        result = self._plan(num_scenes=3)
        for scene in result["scenes"]:
            assert_no_schema_leak(scene["image_prompt"])
            assert_no_schema_leak(scene["video_prompt"])

    def test_unknown_subject_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown subject"):
            sc.resolve_subject("zeus")

    def test_resolve_subject_presets(self):
        self.assertIsNone(sc.resolve_subject("auto"))
        self.assertIsNone(sc.resolve_subject(None))
        god = sc.resolve_subject("god")
        self.assertEqual(god["identity"], "God")
        self.assertEqual(sc.resolve_subject("none")["type"], "none")


class ScriptFlowTest(unittest.TestCase):
    def _ctx(self, tmp):
        import state as st
        state = st.new_state("b")
        return {"project": "b", "base_dir": tmp, "base_seed": 1,
                "state": state,
                "state_path": os.path.join(tmp, "state.json"),
                "storyboard_path": os.path.join(tmp, "storyboard.json")}

    def _args(self, **kwargs):
        base = {"research": "none", "prompt": None, "keyframes": None,
                "script": "bible.txt", "analysis_model": "qwen3:8b",
                "scenes": 3, "subject": "god", "character_reference": None}
        base.update(kwargs)
        return SimpleNamespace(**base)

    def _board(self, n=3):
        return {"research": None, "subject": dict(sc.SUBJECT_PRESETS["god"]),
                "scenes": [{
                    "id": i + 1, "duration": 7,
                    "image_prompt": f"Scene visual {i + 1}.",
                    "video_prompt": f"Motion {i + 1}.",
                    "narration": f"Spoken line {i + 1}."}
                    for i in range(n)]}

    def test_script_path_persists_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("script.generate_from_script",
                        return_value=(self._board(), dict(CONDENSED),
                                      "source words")) as gen, \
                 patch.object(ol, "generate_storyboard",
                              side_effect=AssertionError("text path")), \
                 patch.object(ol, "generate_storyboard_from_keyframes",
                              side_effect=AssertionError("keyframe path")):
                rc = main.run_fresh_flow(
                    self._args(), self._ctx(tmp), {},
                    "http://127.0.0.1:11434", "qwen3:8b", 3, ["storyboard"])
            self.assertEqual(rc, 0)
            gen.assert_called_once()
            model_kw = gen.call_args[1]
            self.assertEqual(model_kw["model"], "qwen3:8b")
            self.assertEqual(
                Path(tmp, "source.txt").read_text(encoding="utf-8"),
                "source words")
            condensed = json.load(
                open(os.path.join(tmp, "condensed_script.json"),
                     encoding="utf-8"))
            self.assertEqual(condensed["title"], "The Shepherd")
            saved = json.load(
                open(os.path.join(tmp, "storyboard.json"), encoding="utf-8"))
            self.assertEqual(len(saved["scenes"]), 3)
            self.assertEqual(saved["subject"]["identity"], "God")

    def test_other_modes_unaffected(self):
        board = self._board(n=1)
        with tempfile.TemporaryDirectory() as tmp:
            base = {"research": "none", "prompt": "idea", "keyframes": None,
                    "script": None, "analysis_model": None, "scenes": None,
                    "subject": "auto", "character_reference": None}
            with patch.object(ol, "generate_storyboard",
                               return_value=board), \
                 patch("script.generate_from_script",
                       side_effect=AssertionError("script path used")):
                rc = main.run_fresh_flow(
                    SimpleNamespace(**base), self._ctx(tmp), {},
                    "http://127.0.0.1:11434", "qwen3:8b", 3, ["storyboard"])
            self.assertEqual(rc, 0)


class ScriptResumeTest(unittest.TestCase):
    def test_resume_never_calls_qwen(self):
        import state as st

        with tempfile.TemporaryDirectory() as tmp:
            board = {"research": None,
                     "subject": dict(sc.SUBJECT_PRESETS["god"]),
                     "scenes": [{"id": 1, "duration": 7,
                                 "image_prompt": "God above a crowd.",
                                 "video_prompt": "Robes move gently.",
                                 "narration": "Voice."}]}
            sb_path = os.path.join(tmp, "storyboard.json")
            with open(sb_path, "w", encoding="utf-8") as f:
                json.dump(board, f)
            import struct
            tiny_png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13)
                        + b"IHDR" + struct.pack(">IIBBBBB", 8, 8, 8, 2, 0,
                                                0, 0) + struct.pack(">I", 0))

            class StubImageProvider:
                def generate(self, req):
                    Path(str(req.output_path)).write_bytes(tiny_png)
                    return SimpleNamespace(
                        path=Path(str(req.output_path)), width=8, height=8)

            ctx = {"project": "b", "storyboard_path": sb_path,
                   "state": st.new_state("b"),
                   "state_path": os.path.join(tmp, "state.json"),
                   "subject": None, "char_ref": None,
                   "image_provider": StubImageProvider(),
                   "image_model": "stub",
                   "img_dir": tmp, "img_neg": "", "target_w": 8,
                   "target_h": 8, "base_seed": 1}
            args = SimpleNamespace(stage="image", scene=1,
                                   character_reference=None)
            with patch("script.generate_from_script",
                        side_effect=AssertionError("Qwen invoked")), \
                 patch.object(ol, "generate_storyboard",
                              side_effect=AssertionError("Ollama invoked")), \
                 patch.object(ol, "generate_storyboard_from_keyframes",
                              side_effect=AssertionError("Qwen-VL invoked")):
                rc = main.run_resume_flow(
                    args, ctx, sb.validate_storyboard(board), ["image"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
