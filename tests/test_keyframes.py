"""Unit tests: Mode B keyframe-to-storyboard (Qwen-VL via Ollama).

No GPU/network: the Ollama transport (ollama._post) and reachability
checks are mocked. The key assertion throughout is that image BYTES are
attached to the vision request (message["images"]), never just filenames.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import ollama as ol
import storyboard as sb
from subject import assert_no_schema_leak

TINY_PNG = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
            + struct.pack(">I", 0))


def make_keyframes(names=("001.jpg", "002.png", "003.webp")):
    tmp = tempfile.TemporaryDirectory()
    for name in names:
        Path(tmp.name, name).write_bytes(TINY_PNG)
    tmp_names = names
    tmp.dir_names = tmp_names
    return tmp


def vision_response(image_prompt="A river through a green valley.",
                    video_prompt="Water flows while the camera pans slowly.",
                    subject=None):
    body = {"image_prompt": image_prompt, "video_prompt": video_prompt}
    if subject is not None:
        body["subject"] = subject
    return {"message": {"content": json.dumps(body)}}


class KeyframeCliTest(unittest.TestCase):
    def test_parses_keyframes(self):
        args = main.parse_args(["--project", "w", "--keyframes", "kf"])
        self.assertEqual(args.keyframes, "kf")

    def test_parses_analysis_model(self):
        args = main.parse_args(
            ["--project", "w", "--keyframes", "kf",
             "--analysis-model", "qwen3-vl:8b"])
        self.assertEqual(args.analysis_model, "qwen3-vl:8b")

    def test_analysis_alias(self):
        args = main.parse_args(
            ["--project", "w", "--keyframes", "kf",
             "--analysis", "qwen2.5vl:7b"])
        self.assertEqual(args.analysis_model, "qwen2.5vl:7b")

    def test_default_model(self):
        args = main.parse_args(["--project", "w", "--keyframes", "kf"])
        self.assertEqual(args.analysis_model, "qwen3-vl:8b")

    def test_prompt_optional_with_keyframes(self):
        args = main.parse_args(["--project", "w", "--keyframes", "kf"])
        self.assertIsNone(args.prompt)
        self.assertIsNone(main.validate_cli_combination(args))

    def test_prompt_required_without_keyframes_or_resume(self):
        args = main.parse_args([])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_prompt_conflicts_with_keyframes(self):
        args = main.parse_args(["idea", "--keyframes", "kf"])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_keyframes_conflicts_with_resume(self):
        args = main.parse_args(
            ["--project", "w", "--resume", "--keyframes", "kf"])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_keyframes_requires_project(self):
        args = main.parse_args(["--keyframes", "kf"])
        self.assertIsNotNone(main.validate_cli_combination(args))


class KeyframeDiscoveryTest(unittest.TestCase):
    def test_all_supported_extensions(self):
        names = ("a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp", "f.tif",
                 "g.tiff")
        with make_keyframes(names) as tmp:
            found = ol.discover_keyframes(tmp)
        self.assertEqual(len(found), len(names))

    def test_case_insensitive(self):
        with make_keyframes(("A.JPG", "B.PNG", "C.WEBP")) as tmp:
            found = ol.discover_keyframes(tmp)
        self.assertEqual(len(found), 3)

    def test_unsupported_ignored(self):
        with make_keyframes(("ok.jpg", "note.txt", "anim.gif",
                             "doc.pdf")) as tmp:
            found = ol.discover_keyframes(tmp)
        self.assertEqual([os.path.basename(p) for p in found], ["ok.jpg"])

    def test_natural_sort(self):
        with make_keyframes(("10.png", "2.png", "1.png", "20.jpg")) as tmp:
            found = [os.path.basename(p)
                     for p in ol.discover_keyframes(tmp)]
        self.assertEqual(found, ["1.png", "2.png", "10.png", "20.jpg"])

    def test_empty_directory_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "No supported keyframe"):
                ol.discover_keyframes(tmp)

    def test_missing_directory_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            ol.discover_keyframes("/nonexistent/keyframes/dir")


class KeyframeImagePayloadTest(unittest.TestCase):
    def test_images_field_carries_file_bytes(self):
        with make_keyframes(("frame.png",)) as tmp:
            path = os.path.join(tmp, "frame.png")
            expected_b64 = base64.b64encode(
                Path(path).read_bytes()).decode("ascii")
            captured = {}

            def fake_post(url, payload, timeout):
                captured.update(payload)
                return vision_response()

            with patch.object(ol, "_post", side_effect=fake_post), \
                 patch.object(ol, "check_model_available",
                              return_value=None):
                result = ol.generate_storyboard_from_keyframes(
                    tmp, model="qwen3-vl:8b", num_scenes=1)
        images = captured["messages"][1].get("images")
        self.assertIsNotNone(images, "vision request lacks images field")
        self.assertEqual(images, [expected_b64])
        self.assertEqual(len(result["scenes"]), 1)

    def test_load_keyframe_b64_roundtrip(self):
        with make_keyframes(("a.jpg",)) as tmp:
            path = os.path.join(tmp, "a.jpg")
            self.assertEqual(base64.b64decode(ol.load_keyframe_b64(path)),
                             Path(path).read_bytes())

    def test_empty_image_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.png")
            Path(path).write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                ol.load_keyframe_b64(path)


class KeyframeAnalysisTest(unittest.TestCase):
    def test_fence_stripped(self):
        fenced = ('```json\n{"image_prompt": "A lake.", '
                  '"video_prompt": "Ripples move slowly."}\n```')
        with patch.object(ol, "_post",
                           return_value={"message": {"content": fenced}}):
            out = ol.analyze_keyframe("aGk=", model="m", label="k")
        self.assertEqual(out["image_prompt"], "A lake.")

    def test_malformed_json_bounded_retry_then_fail(self):
        bad = {"message": {"content": "not json at all"}}
        with patch.object(ol, "_post", return_value=bad) as post:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                ol.analyze_keyframe("aGk=", model="m", label="kf001.png")
        self.assertEqual(post.call_count, 3)

    def test_transient_malformed_then_success(self):
        calls = {"n": 0}

        def flaky(url, payload, timeout):
            calls["n"] += 1
            if calls["n"] < 2:
                return {"message": {"content": "garbage {"}}
            return vision_response()

        with patch.object(ol, "_post", side_effect=flaky):
            out = ol.analyze_keyframe("aGk=", model="m", label="k")
        self.assertEqual(calls["n"], 2)
        self.assertIn("valley", out["image_prompt"])

    def test_schema_leak_retried_then_fails(self):
        leaky = vision_response(image_prompt="Scene: a lake at dawn")
        with patch.object(ol, "_post", return_value=leaky) as post:
            with self.assertRaisesRegex(RuntimeError, "kf002.png"):
                ol.analyze_keyframe("aGk=", model="m", label="kf002.png")
        self.assertEqual(post.call_count, 3)

    def test_model_unavailable_names_pull_command(self):
        tags = {"models": [{"name": "qwen3:8b"}]}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(tags).encode()

        with patch("urllib.request.urlopen", return_value=FakeResp()):
            with self.assertRaisesRegex(RuntimeError, "ollama pull qwen3-vl"):
                ol.check_model_available("http://x", "qwen3-vl:8b")

    def test_ollama_unreachable_is_clear(self):
        with patch("urllib.request.urlopen",
                   side_effect=OSError("connection refused")):
            with self.assertRaisesRegex(RuntimeError, "not reachable"):
                ol.check_model_available("http://x", "qwen3-vl:8b")


class KeyframeStoryboardTest(unittest.TestCase):
    def _generate(self, tmp, **kwargs):
        with patch.object(ol, "_post",
                          return_value=vision_response()), \
             patch.object(ol, "check_model_available", return_value=None):
            return ol.generate_storyboard_from_keyframes(tmp, **kwargs)

    def test_one_keyframe_one_scene(self):
        with make_keyframes(("only.png",)) as tmp:
            result = self._generate(tmp, model="m")
        self.assertEqual(len(result["scenes"]), 1)
        self.assertEqual(result["scenes"][0]["id"], 1)

    def test_three_keyframes_three_scenes(self):
        with make_keyframes() as tmp:
            result = self._generate(tmp, model="m")
        self.assertEqual([s["id"] for s in result["scenes"]], [1, 2, 3])

    def test_scenes_truncated_to_n(self):
        with make_keyframes() as tmp:
            result = self._generate(tmp, model="m", num_scenes=2)
        self.assertEqual(len(result["scenes"]), 2)

    def test_scenes_beyond_count_fails(self):
        with make_keyframes(("a.png",)) as tmp:
            with self.assertRaisesRegex(ValueError, "only 1 keyframes"):
                self._generate(tmp, model="m", num_scenes=5)

    def test_schema_shape(self):
        with make_keyframes() as tmp:
            result = self._generate(tmp, model="m")
        self.assertIsNone(result["research"])
        self.assertEqual(result["subject"], {
            "type": "none", "identity": "", "features": "",
            "clothing_or_equipment": "", "consistency": ""})
        for scene in result["scenes"]:
            self.assertEqual(scene["duration"], 7)
            self.assertEqual(scene["narration"], "")
            self.assertTrue(scene["image_prompt"].strip())
            self.assertTrue(scene["video_prompt"].strip())
        validated = sb.validate_storyboard(result)
        self.assertEqual(len(validated), 3)

    def test_scenery_subject_defaults_none(self):
        with make_keyframes() as tmp:
            result = self._generate(tmp, model="m")
        self.assertEqual(result["subject"]["type"], "none")

    def test_unanimous_person_subject_kept(self):
        person = {"type": "person", "identity": "a hiker",
                  "features": "red jacket", "clothing_or_equipment": "",
                  "consistency": "same hiker"}
        with patch.object(ol, "_post",
                           return_value=vision_response(subject=person)), \
             patch.object(ol, "check_model_available", return_value=None):
            with make_keyframes() as tmp:
                result = ol.generate_storyboard_from_keyframes(tmp, model="m")
        self.assertEqual(result["subject"]["type"], "person")

    def test_prompts_pass_leak_guard(self):
        with make_keyframes() as tmp:
            result = self._generate(tmp, model="m")
        for scene in result["scenes"]:
            assert_no_schema_leak(scene["image_prompt"])
            assert_no_schema_leak(scene["video_prompt"])


class KeyframeFlowTest(unittest.TestCase):
    def _ctx(self, tmp):
        import state as st
        state = st.new_state("w")
        state_path = os.path.join(tmp, "state.json")
        return {"project": "w", "base_seed": 1, "state": state,
                "state_path": state_path,
                "storyboard_path": os.path.join(tmp, "storyboard.json")}

    def _args(self, **kwargs):
        base = {"research": "none", "prompt": None, "keyframes": None,
                "analysis_model": "qwen3-vl:8b", "scenes": None,
                "character_reference": None}
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_keyframe_path_saves_storyboard(self):
        board = {"research": None,
                 "subject": {"type": "none", "identity": "", "features": "",
                             "clothing_or_equipment": "", "consistency": ""},
                 "scenes": [{"id": 1, "duration": 7, "image_prompt": "A.",
                             "video_prompt": "B.", "narration": ""}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ol, "generate_storyboard_from_keyframes",
                               return_value=board) as gen, \
                 patch.object(ol, "generate_storyboard",
                              side_effect=AssertionError("text path used")):
                rc = main.run_fresh_flow(
                    self._args(keyframes="kfdir"), self._ctx(tmp), {},
                    "http://127.0.0.1:11434", "qwen3:8b", 3, ["storyboard"])
            self.assertEqual(rc, 0)
            gen.assert_called_once()
            saved = json.load(
                open(os.path.join(tmp, "storyboard.json"), encoding="utf-8"))
            self.assertEqual(len(saved["scenes"]), 1)

    def test_text_path_unaffected(self):
        board = {"research": None,
                 "subject": {"type": "none", "identity": "", "features": "",
                             "clothing_or_equipment": "", "consistency": ""},
                 "scenes": [{"id": 1, "duration": 5, "image_prompt": "A.",
                             "video_prompt": "B.", "narration": "Voice."}]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ol, "generate_storyboard",
                               return_value=board) as gen, \
                 patch.object(ol, "generate_storyboard_from_keyframes",
                              side_effect=AssertionError("keyframe path")):
                rc = main.run_fresh_flow(
                    self._args(prompt="idea"), self._ctx(tmp), {},
                    "http://127.0.0.1:11434", "qwen3:8b", 3, ["storyboard"])
            self.assertEqual(rc, 0)
            gen.assert_called_once()


class KeyframeResumeTest(unittest.TestCase):
    def test_resume_never_calls_vision(self):
        import state as st

        with tempfile.TemporaryDirectory() as tmp:
            board = {"research": None,
                     "subject": {"type": "none", "identity": "",
                                 "features": "", "clothing_or_equipment": "",
                                 "consistency": ""},
                     "scenes": [{"id": 1, "duration": 5,
                                 "image_prompt": "A calm lake at dawn.",
                                 "video_prompt": "Mist drifts slowly.",
                                 "narration": "Voice."}]}
            sb_path = os.path.join(tmp, "storyboard.json")
            with open(sb_path, "w", encoding="utf-8") as f:
                json.dump(board, f)

            class StubImageProvider:
                def generate(self, req):
                    Path(str(req.output_path)).write_bytes(TINY_PNG)
                    return SimpleNamespace(
                        path=Path(str(req.output_path)), width=8, height=8)

            ctx = {"project": "w", "storyboard_path": sb_path,
                   "state": st.new_state("w"),
                   "state_path": os.path.join(tmp, "state.json"),
                   "subject": None, "char_ref": None,
                   "image_provider": StubImageProvider(),
                   "image_model": "stub", "img_dir": tmp,
                   "img_neg": "", "target_w": 8, "target_h": 8,
                   "base_seed": 1}
            args = SimpleNamespace(stage="image", scene=1,
                                   character_reference=None)
            with patch.object(
                    ol, "generate_storyboard_from_keyframes",
                    side_effect=AssertionError("Qwen-VL invoked")), \
                 patch.object(ol, "generate_storyboard",
                              side_effect=AssertionError("Ollama invoked")):
                rc = main.run_resume_flow(
                    args, ctx, sb.validate_storyboard(board), ["image"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
