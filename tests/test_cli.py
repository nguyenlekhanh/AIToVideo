"""Unit tests: CLI parsing (new flags + backward compat), startup validation."""
from __future__ import annotations

import os
import unittest

import main
from providers.errors import UnknownModelError
from providers.registry import lookup

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CliTest(unittest.TestCase):
    def test_new_flags(self):
        args = main.parse_args(["hello", "--project", "mars_city",
                                "--image-model", "flux", "--video-model", "ltx",
                                "--aspect", "9:16", "--resolution", "720"])
        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.project, "mars_city")
        self.assertEqual(args.image_model, "flux")
        self.assertEqual(args.video_model, "ltx")
        self.assertEqual(args.aspect, "9:16")
        self.assertEqual(args.resolution, "720")

    def test_backward_compat_defaults(self):
        # Old command without any new flags still parses and maps to
        # the original pipeline (sd15 keyframes + ltx video).
        args = main.parse_args(["A lone astronaut discovers an ancient city on Mars"])
        self.assertEqual(args.image_model, "sd15")
        self.assertEqual(args.video_model, "ltx")
        self.assertEqual(args.audio_model, "edge_tts")
        self.assertEqual(args.aspect, "16:9")
        self.assertEqual(args.resolution, 720)

    def test_backward_compat_old_flags(self):
        args = main.parse_args(["hello", "--model", "qwen3:14b",
                                "--voice", "vi-VN-HoaiMyNeural",
                                "--project", "mars_city", "--seed", "7"])
        self.assertEqual(args.model, "qwen3:14b")
        self.assertEqual(args.voice, "vi-VN-HoaiMyNeural")
        self.assertEqual(args.project, "mars_city")
        self.assertEqual(args.seed, 7)

    def test_unknown_model_rejected_by_registry(self):
        with self.assertRaises(UnknownModelError):
            lookup(AI_VIDEO_DIR, "image", "krea2")
        with self.assertRaises(UnknownModelError):
            lookup(AI_VIDEO_DIR, "video", "sora")

    def test_krea_model_selected(self):
        args = main.parse_args(["hello", "--image-model", "krea",
                                "--video-model", "ltx",
                                "--aspect", "9:16", "--resolution", "720"])
        self.assertEqual(args.image_model, "krea")
        entry = lookup(AI_VIDEO_DIR, "image", args.image_model)
        self.assertEqual(entry["provider"], "krea")
        self.assertTrue(os.path.isfile(entry["workflow_path"]))

    def test_character_reference_flag(self):
        args = main.parse_args(["hello", "--character-reference", "char.png"])
        self.assertEqual(args.character_reference, "char.png")
        args = main.parse_args(["hello"])
        self.assertIsNone(args.character_reference)

    def test_stop_after_flag(self):
        args = main.parse_args(["hello"])
        self.assertEqual(args.stop_after, "all")
        args = main.parse_args(["hello", "--stop-after", "images"])
        self.assertEqual(args.stop_after, "images")
        for value in ("storyboard", "image", "images", "video", "videos",
                      "audio", "all"):
            with self.subTest(value=value):
                self.assertEqual(
                    main.parse_args(["hello", "--stop-after", value]).stop_after,
                    value)
        with self.assertRaises(SystemExit):
            main.parse_args(["hello", "--stop-after", "bogus"])

    def test_resume_stage_scene_flags(self):
        args = main.parse_args(["--project", "mars_city1", "--resume"])
        self.assertTrue(args.resume)
        self.assertIsNone(args.prompt)
        self.assertIsNone(args.stage)
        self.assertIsNone(args.scene)
        args = main.parse_args(["--project", "mars_city1", "--resume",
                                "--stage", "video", "--scene", "3"])
        self.assertEqual(args.stage, "video")
        self.assertEqual(args.scene, 3)
        with self.assertRaises(SystemExit):
            main.parse_args(["hello", "--stage", "bogus"])
        # prompt optional only with --resume
        self.assertIsNone(main.parse_args(
            ["--project", "x", "--resume"]).prompt)

    def test_cli_combination_validation(self):
        ok = [
            ["hello"],
            ["hello", "--stop-after", "images"],
            ["--project", "x", "--resume"],
            ["--project", "x", "--resume", "--stage", "video"],
            ["--project", "x", "--resume", "--stage", "image", "--scene", "3"],
            ["--project", "x", "--resume", "--scene", "2"],
        ]
        for argv in ok:
            with self.subTest(argv=argv):
                self.assertIsNone(
                    main.validate_cli_combination(main.parse_args(argv)))
        bad = [
            ([], "prompt required"),
            (["--resume"], "project required"),
            (["hello", "--stage", "video"], "stage needs resume"),
            (["hello", "--scene", "3"], "scene needs resume"),
            (["--project", "x", "--resume", "--scene", "0"], "positive scene"),
            (["--project", "x", "--resume", "--stage", "video",
              "--stop-after", "video"], "mutually exclusive"),
        ]
        for argv, _label in bad:
            with self.subTest(argv=argv):
                self.assertIsNotNone(
                    main.validate_cli_combination(main.parse_args(argv)))
        # prompt missing without resume covered via parse (prompt optional)
        self.assertIsNotNone(main.validate_cli_combination(
            main.parse_args([])))

    def test_registered_models_exist_on_disk(self):
        for kind, model in (("image", "sd15"), ("image", "sdxl"), ("image", "flux"),
                            ("image", "krea"), ("video", "ltx")):
            with self.subTest(kind=kind, model=model):
                entry = lookup(AI_VIDEO_DIR, kind, model)
                self.assertTrue(os.path.isfile(entry["workflow_path"]),
                                f"Missing workflow: {entry.get('workflow_path')}")


if __name__ == "__main__":
    unittest.main()
