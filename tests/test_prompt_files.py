"""Unit tests: project-local sequential video prompt files (mocked, no GPU)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import promptfiles as pf
import state as st


def write_prompt(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


MULTI_SCENE_PROMPT = """Create a 10-second vertical 9:16 breaking-news style video.

SCENE 1 -- 0-2 seconds:
A dramatic modern television news studio.

Display a large, clear headline:
"TRUMP NAMES LA & SAN DIEGO"

SCENE 2 -- 2-5 seconds:
Cut to a fictional politician at a podium.
No spoken dialogue.

IMPORTANT:
No real person likeness.
No spoken dialogue.
No voice-over.
No real news footage.
"""


class DiscoveryTest(unittest.TestCase):
    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", "River dawn.")
            entries, ignored = pf.discover_prompt_files(tmp)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0][0], 1)
            self.assertEqual(ignored, [])

    def test_natural_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "10.txt", "11.txt", "3.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            entries, _ = pf.discover_prompt_files(tmp)
            self.assertEqual([n for n, _ in entries], [1, 2, 3, 10, 11])

    def test_leading_zeros_and_uppercase_ext(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("003.txt", "004.txt", "10.TXT"):
                write_prompt(tmp, name, f"Prompt {name}.")
            entries, _ = pf.discover_prompt_files(tmp)
            self.assertEqual([n for n, _ in entries], [3, 4, 10])

    def test_arbitrary_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(1, 21):
                write_prompt(tmp, f"{i}.txt", f"Prompt {i}.")
            entries, _ = pf.discover_prompt_files(tmp)
            self.assertEqual([n for n, _ in entries], list(range(1, 21)))

    def test_unrelated_files_ignored_with_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", "One.")
            write_prompt(tmp, "2.txt", "Two.")
            write_prompt(tmp, "abc.txt", "Not a clip.")
            write_prompt(tmp, "scene1.txt", "Not a clip.")
            write_prompt(tmp, "1_backup.txt", "Not a clip.")
            write_prompt(tmp, "notes.md", "Not a prompt.")
            entries, ignored = pf.discover_prompt_files(tmp)
            self.assertEqual([n for n, _ in entries], [1, 2])
            self.assertEqual(sorted(ignored),
                             ["1_backup.txt", "abc.txt", "scene1.txt"])

    def test_gap_allowed_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "4.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            self.assertEqual([s["id"] for s in scenes], [1, 2, 4])

    def test_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                pf.discover_prompt_files(os.path.join(tmp, "prompt"))

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                pf.discover_prompt_files(tmp)
            self.assertIn("No video prompt files", str(ctx.exception))

    def test_only_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "notes.md", "x")
            with self.assertRaises(ValueError):
                pf.discover_prompt_files(tmp)

    def test_empty_prompt_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_prompt(tmp, "1.txt", "   \n  ")
            with self.assertRaises(ValueError) as ctx:
                pf.read_prompt_text(path)
            self.assertIn("1.txt", str(ctx.exception))

    def test_duplicate_clip_number_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "3.txt", "Three.")
            write_prompt(tmp, "003.txt", "Three again.")
            with self.assertRaises(ValueError) as ctx:
                pf.discover_prompt_files(tmp)
            self.assertIn("3", str(ctx.exception))

    def test_utf8_vietnamese_preserved(self):
        text = "Cảnh sông Hương lúc bình minh, sương mù giăng trên mặt nước."
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", text)
            self.assertEqual(pf.read_prompt_text(
                os.path.join(tmp, "1.txt")), text)

    def test_file_level_whitespace_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", "\n\n  Keep this.\nLine two.\n\n")
            self.assertEqual(pf.read_prompt_text(
                os.path.join(tmp, "1.txt")), "Keep this.\nLine two.")


class PreservationTest(unittest.TestCase):
    def test_multi_scene_text_is_one_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", MULTI_SCENE_PROMPT)
            write_prompt(tmp, "2.txt", "Second clip.")
            scenes = pf.load_prompt_scenes(tmp, 10.0)
            self.assertEqual(len(scenes), 2)
            self.assertEqual(scenes[0]["video_prompt"], MULTI_SCENE_PROMPT.strip())
            self.assertEqual(scenes[0]["duration"], 10.0)
            self.assertEqual(scenes[0]["narration"], "")
            # Embedded SCENE headings are model instructions, not scenes.
            self.assertNotIn("SCENE", str(scenes[0]["id"]))

    def test_scene_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "3.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            for scene, expect in zip(scenes, (1, 2, 3)):
                self.assertEqual(scene["id"], expect)
                self.assertEqual(scene["prompt_file"], f"{expect}.txt")


class PipelineTest(unittest.TestCase):
    class StubProvider:
        needs_input_image = False

        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            Path(str(request.output_path)).write_bytes(b"fakevideo")
            return SimpleNamespace(path=Path(str(request.output_path)),
                                   width=1312, height=736)

    def _ctx(self, tmp):
        state = st.new_state("w")
        return {"project": "w", "base_dir": tmp, "base_seed": 1,
                "state": state, "state_path": os.path.join(tmp, "s.json"),
                "img_dir": tmp, "vid_dir": tmp, "video_provider": self.StubProvider(),
                "video_model": "ltx", "vid_neg": "",
                "target_w": 1312, "target_h": 736, "default_duration": 10.0,
                "prompt_mode": True}

    def _scenes(self, tmp):
        write_prompt(tmp, "1.txt", MULTI_SCENE_PROMPT)
        write_prompt(tmp, "2.txt", "Second clip.")
        return pf.load_prompt_scenes(tmp, 10.0)

    def test_verbatim_prompt_per_clip_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            scenes = self._scenes(tmp)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                paths = main.run_video_stage(ctx, scenes, {}, force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 2)
            # Whole file = one prompt; SCENE headings never split the job.
            self.assertEqual(prov.requests[0].prompt, MULTI_SCENE_PROMPT.strip())
            self.assertIn("SCENE 2", prov.requests[0].prompt)
            self.assertEqual(prov.requests[1].prompt, "Second clip.")
            for req in prov.requests:
                self.assertIsNone(req.input_image)
                self.assertEqual(req.duration, 10.0)
            self.assertEqual(len(paths), 2)
            self.assertTrue(str(paths[1]).endswith("scene_001.mp4"))
            self.assertTrue(str(paths[2]).endswith("scene_002.mp4"))

    def test_resume_skips_valid_regenerates_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            scenes = self._scenes(tmp)
            seen = {"n": 0}

            def fake_valid(path, *args):
                s = str(path)
                if s.endswith(".tmp"):
                    return True  # post-generation validation of fresh output
                # scene_001 valid, scene_002 invalid
                return "scene_001.mp4" in s

            with patch.object(st, "valid_video_file", side_effect=fake_valid), \
                 patch.object(st, "valid_image_file", return_value=False):
                paths = main.run_video_stage(ctx, scenes, {}, force_all=False)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 1)
            self.assertEqual(prov.requests[0].prompt, "Second clip.")
            self.assertEqual(len(paths), 2)

    def test_scene_selection_processes_only_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            scenes = self._scenes(tmp)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, scenes, {}, only_scene=2)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 1)
            self.assertEqual(prov.requests[0].prompt, "Second clip.")

    def test_no_storyboard_planner_or_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            scenes = self._scenes(tmp)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False), \
                 patch("ollama.generate_storyboard",
                       side_effect=AssertionError("planner called")), \
                 patch("providers.research.web.WebResearchProvider.research",
                       side_effect=AssertionError("research called")):
                main.run_video_stage(ctx, scenes, {}, force_all=True)
            self.assertFalse(os.path.isfile(os.path.join(tmp, "storyboard.json")))

    def test_state_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            scenes = self._scenes(tmp)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, scenes, {}, force_all=True)
            saved = st.load_state(ctx["state_path"], "w")
            self.assertEqual(
                saved["stages"]["video"]["completed_scenes"], [1, 2])


class CliTest(unittest.TestCase):
    def _args(self, *argv):
        return main.parse_args(list(argv))

    def test_prompt_dir_parses(self):
        args = self._args("--project", "news1", "--resume", "--stage", "video",
                          "--video-model", "ltx", "--video-mode", "t2v",
                          "--prompt-dir", "prompt")
        self.assertEqual(args.prompt_dir, "prompt")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_prompt_dir_requires_project(self):
        args = self._args("--resume", "--stage", "video",
                          "--prompt-dir", "prompt")
        self.assertIn("--project", main.validate_cli_combination(args))

    def test_prompt_dir_rejects_positional_prompt(self):
        args = self._args("some idea", "--project", "news1",
                          "--prompt-dir", "prompt")
        err = main.validate_cli_combination(args)
        self.assertIsNotNone(err)

    def test_prompt_dir_rejects_keyframes_script(self):
        for extra in (("--keyframes", "k"), ("--script", "s.txt")):
            args = self._args("--project", "news1", "--prompt-dir", "prompt",
                              *extra)
            self.assertIsNotNone(main.validate_cli_combination(args))

    def test_prompt_dir_rejects_scenes_and_image_stage(self):
        args = self._args("--project", "news1", "--prompt-dir", "prompt",
                          "--scenes", "3")
        self.assertIsNotNone(main.validate_cli_combination(args))
        args = self._args("--project", "news1", "--resume", "--stage", "image",
                          "--prompt-dir", "prompt")
        self.assertIn("image", main.validate_cli_combination(args))

    def test_prompt_fresh_needs_no_positional_prompt(self):
        args = self._args("--project", "news1", "--prompt-dir", "prompt")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_existing_flows_unaffected(self):
        args = self._args("a video idea", "--project", "p1")
        self.assertIsNone(main.validate_cli_combination(args))
        args = self._args("--project", "p1", "--resume", "--stage", "video",
                          "--video-model", "ltx", "--video-mode", "t2v")
        self.assertIsNone(main.validate_cli_combination(args))


def write_manifest(tmp: str, clips: dict) -> str:
    import json
    path = os.path.join(tmp, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"clips": clips}, f)
    return path


class ManifestTest(unittest.TestCase):
    def _three(self, tmp):
        for name in ("1.txt", "2.txt", "3.txt"):
            write_prompt(tmp, name, f"Prompt {name}.")

    def test_no_manifest_version1_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            self.assertIsNone(pf.load_manifest(tmp))
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            self.assertEqual([s["duration"] for s in scenes], [5.0] * 3)

    def test_valid_manifest_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10},
                                 "3": {"duration": 10}})
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            self.assertEqual([(s["id"], s["duration"]) for s in scenes],
                             [(1, 10), (2, 10), (3, 10)])
            # Pipeline default is ignored when the manifest exists.
            self.assertNotIn(5.0, [s["duration"] for s in scenes])

    def test_per_clip_durations_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            write_manifest(tmp, {"1": {"duration": 5},
                                 "2": {"duration": 6},
                                 "3": {"duration": 7}})
            scenes = pf.load_prompt_scenes(tmp, 9.0)
            self.assertEqual([s["duration"] for s in scenes], [5, 6, 7])

    def test_prompt_verbatim_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", MULTI_SCENE_PROMPT)
            write_manifest(tmp, {"1": {"duration": 10}})
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            self.assertEqual(scenes[0]["video_prompt"],
                             MULTI_SCENE_PROMPT.strip())
            self.assertIn("SCENE 2", scenes[0]["video_prompt"])

    def test_manifest_outside_prompt_dir_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = os.path.join(tmp, "prompt")
            os.makedirs(prompt_dir)
            self._three(prompt_dir)
            import json
            with open(os.path.join(tmp, "manifest.json"), "w",
                       encoding="utf-8") as f:
                json.dump({"clips": {"1": {"duration": 99}}}, f)
            self.assertIsNone(pf.load_manifest(prompt_dir))
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            self.assertEqual([s["duration"] for s in scenes], [5.0] * 3)

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            with open(os.path.join(tmp, "manifest.json"), "w",
                       encoding="utf-8") as f:
                f.write("{not json")
            with self.assertRaises(ValueError) as ctx:
                pf.load_prompt_scenes(tmp, 5.0)
            self.assertIn("manifest.json", str(ctx.exception).lower())

    def test_missing_clips_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            import json
            with open(os.path.join(tmp, "manifest.json"), "w",
                       encoding="utf-8") as f:
                json.dump({"durations": {}}, f)
            with self.assertRaises(ValueError):
                pf.load_prompt_scenes(tmp, 5.0)

    def test_missing_entry_for_discovered_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10}})
            with self.assertRaises(ValueError) as ctx:
                pf.load_prompt_scenes(tmp, 5.0)
            self.assertIn("clip 3", str(ctx.exception))

    def test_extra_manifest_entry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._three(tmp)
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10},
                                 "3": {"duration": 10},
                                 "4": {"duration": 10}})
            with self.assertRaises(ValueError) as ctx:
                pf.load_prompt_scenes(tmp, 5.0)
            self.assertIn("4", str(ctx.exception))

    def test_non_numeric_clip_key(self):
        for bad in ("one", "1.5", "-2", ""):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                self._three(tmp)
                write_manifest(tmp, {"1": {"duration": 10},
                                     "2": {"duration": 10},
                                     "3": {"duration": 10}, bad: {"duration": 10}})
                with self.assertRaises(ValueError):
                    pf.load_prompt_scenes(tmp, 5.0)

    def test_invalid_duration_types(self):
        for bad in ("10 seconds", "10s", "ten", None, True, [10], {"v": 10}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                write_prompt(tmp, "1.txt", "One.")
                write_manifest(tmp, {"1": {"duration": bad}})
                with self.assertRaises(ValueError):
                    pf.load_prompt_scenes(tmp, 5.0)

    def test_invalid_duration_values(self):
        for bad in (0, -5, float("nan"), float("inf")):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                write_prompt(tmp, "1.txt", "One.")
                write_manifest(tmp, {"1": {"duration": bad}})
                with self.assertRaises(ValueError):
                    pf.load_prompt_scenes(tmp, 5.0)

    def test_missing_duration_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_prompt(tmp, "1.txt", "One.")
            write_manifest(tmp, {"1": {}})
            with self.assertRaises(ValueError):
                pf.load_prompt_scenes(tmp, 5.0)


class ManifestPipelineTest(unittest.TestCase):
    class StubProvider:
        needs_input_image = False

        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            Path(str(request.output_path)).write_bytes(b"fakevideo")
            return SimpleNamespace(path=Path(str(request.output_path)),
                                   width=1312, height=736)

    def _ctx(self, tmp):
        state = st.new_state("w")
        return {"project": "w", "base_dir": tmp, "base_seed": 1,
                "state": state, "state_path": os.path.join(tmp, "s.json"),
                "img_dir": tmp, "vid_dir": tmp, "video_provider": self.StubProvider(),
                "video_model": "ltx", "vid_neg": "",
                "target_w": 1312, "target_h": 736, "default_duration": 5.0,
                "prompt_mode": True}

    def test_manifest_durations_reach_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "3.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10},
                                 "3": {"duration": 10}})
            ctx = self._ctx(tmp)
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, scenes, {}, force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual([r.duration for r in prov.requests], [10, 10, 10])
            self.assertEqual(prov.requests[0].prompt, "Prompt 1.txt.")

    def test_scene_two_uses_manifest_duration_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "3.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10},
                                 "3": {"duration": 10}})
            ctx = self._ctx(tmp)
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, scenes, {}, only_scene=2)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 1)
            self.assertEqual(prov.requests[0].prompt, "Prompt 2.txt.")
            self.assertEqual(prov.requests[0].duration, 10)

    def test_resume_skips_valid_with_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt", "3.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            write_manifest(tmp, {"1": {"duration": 10},
                                 "2": {"duration": 10},
                                 "3": {"duration": 10}})
            ctx = self._ctx(tmp)
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            with patch.object(st, "valid_video_file", return_value=True), \
                 patch.object(st, "valid_image_file", return_value=False):
                paths = main.run_video_stage(ctx, scenes, {}, force_all=False)
            self.assertEqual(len(ctx["video_provider"].requests), 0)
            self.assertEqual(len(paths), 3)

    def test_version1_regression_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("1.txt", "2.txt"):
                write_prompt(tmp, name, f"Prompt {name}.")
            ctx = self._ctx(tmp)
            scenes = pf.load_prompt_scenes(tmp, 5.0)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, scenes, {}, force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual([r.duration for r in prov.requests], [5.0, 5.0])


if __name__ == "__main__":
    unittest.main()
