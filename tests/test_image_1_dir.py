"""Unit tests: optional clip-1 initial image --image-1-dir (mocked, no GPU)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ffmpeg as ffmod
import main
import promptfiles as pf
import state as st
from providers.comfy import ComfyClient
from providers.registry import lookup
from providers.video.minimax_h3 import MiniMaxH3VideoProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubClient(ComfyClient):
    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.queued = []
        self.uploads = []

    def run(self, workflow, output_keys, dest_path: str) -> str:
        self.queued.append(workflow)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"fake")
        return dest_path

    def upload_image(self, image_path: str) -> str:
        self.uploads.append(str(image_path))
        return os.path.basename(str(image_path))


def make_provider():
    entry = lookup(AI_VIDEO_DIR, "video", "minimax_h3")
    return MiniMaxH3VideoProvider(name="minimax_h3", settings=dict(entry),
                                  client=StubClient())


def make_prompt_dir(tmp, *texts):
    d = os.path.join(tmp, "prompt")
    os.makedirs(d)
    for i, text in enumerate(texts, start=1):
        with open(os.path.join(d, f"{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(d, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"clips": {str(i): {"duration": 10}
                             for i in range(1, len(texts) + 1)}}, f)
    return d


def make_image_dir(tmp, *names):
    d = os.path.join(tmp, "image_start")
    os.makedirs(d)
    for name in names:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return d


def make_ctx(tmp, provider, image_1_dir=None, chain=False):
    state = st.new_state("w")
    return {"project": "w", "base_dir": tmp, "base_seed": 1,
            "state": state, "state_path": os.path.join(tmp, "s.json"),
            "img_dir": tmp, "vid_dir": tmp, "video_provider": provider,
            "video_model": "minimax_h3", "vid_neg": "",
            "target_w": 480, "target_h": 864, "default_duration": 5.0,
            "prompt_mode": True, "first_frame_chain": chain,
            "image_1_dir": image_1_dir, "ffmpeg_exe": "ffmpeg"}


def fake_extract(video_path, out_path, executable="ffmpeg"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"png")
    return out_path


def run_stage(ctx, scenes, valid=True, **kwargs):
    if isinstance(valid, bool):
        vf = patch.object(st, "valid_video_file", return_value=valid)
    else:
        vf = patch.object(st, "valid_video_file", side_effect=valid)
    with vf, patch.object(st, "valid_image_file", return_value=False), \
         patch.object(ffmod, "extract_last_frame", side_effect=fake_extract):
        return main.run_video_stage(ctx, scenes, {}, **kwargs)


class DiscoveryTest(unittest.TestCase):
    def test_discovers_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_image_dir(tmp, "1.png")
            self.assertEqual(pf.discover_initial_image(d),
                             os.path.join(d, "1.png"))

    def test_supported_extensions(self):
        for name in ("1.jpg", "1.jpeg", "1.webp", "1.PNG", "1.JpG"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                d = make_image_dir(tmp, name)
                self.assertEqual(pf.discover_initial_image(d),
                                 os.path.join(d, name))

    def test_unrelated_names_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_image_dir(tmp, "image1.png", "scene_001.png",
                               "1_backup.png", "10.png", "2.png", "notes.txt")
            self.assertIsNone(pf.discover_initial_image(d))

    def test_empty_directory_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_image_dir(tmp)
            self.assertIsNone(pf.discover_initial_image(d))

    def test_multiple_matches_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_image_dir(tmp, "1.png", "1.jpg")
            with self.assertRaises(ValueError) as err:
                pf.discover_initial_image(d)
            text = str(err.exception)
            self.assertIn("1.png", text)
            self.assertIn("1.jpg", text)

    def test_leading_zero_stem_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_image_dir(tmp, "001.png")
            self.assertEqual(pf.discover_initial_image(d),
                             os.path.join(d, "001.png"))


class CliTest(unittest.TestCase):
    def test_defaults_unset(self):
        args = main.parse_args(["--project", "w", "--prompt-dir", "prompt"])
        self.assertIsNone(args.image_1_dir)

    def test_valid_combination(self):
        args = main.parse_args(["--project", "w", "--prompt-dir", "prompt",
                                "--video-model", "minimax_h3",
                                "--image-1-dir", "image_start"])
        self.assertIsNone(main.validate_cli_combination(args))

    def test_requires_prompt_dir(self):
        args = main.parse_args(["--project", "w", "--video-model",
                                "minimax_h3", "--image-1-dir", "image_start"])
        self.assertIn("--prompt-dir", main.validate_cli_combination(args))

    def test_rejects_unsupported_backend(self):
        for model in ("wan", "wan_animate2"):
            args = main.parse_args(["--project", "w", "--prompt-dir", "prompt",
                                    "--video-model", model,
                                    "--image-1-dir", "image_start"])
            self.assertIn("minimax_h3", main.validate_cli_combination(args))

    def test_accepts_ltx_family(self):
        for model in ("ltx", "ltx_t2v"):
            args = main.parse_args(["--project", "w", "--prompt-dir", "prompt",
                                    "--video-model", model,
                                    "--image-1-dir", "image_start"])
            self.assertIsNone(main.validate_cli_combination(args))

    def test_missing_directory_fails_before_comfy(self):
        import contextlib
        import io
        import shutil
        proj = "img1_missing_dir_tmp"
        base = os.path.join(AI_VIDEO_DIR, "projects", proj)
        self.addCleanup(shutil.rmtree, base, True)
        os.makedirs(os.path.join(base, "prompt"))
        with open(os.path.join(base, "prompt", "1.txt"), "w",
                  encoding="utf-8") as f:
            f.write("One.")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main.main(["--project", proj,
                              "--video-model", "minimax_h3",
                              "--prompt-dir", "prompt",
                              "--image-1-dir", "nope",
                              "--stop-after", "video"])
        self.assertEqual(code, 1)
        self.assertIn("--image-1-dir not found", err.getvalue())


class StageTest(unittest.TestCase):
    def test_clip1_uses_image_clip2_t2v_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 2)
            first, second = prov.client.queued
            self.assertEqual(first["1"]["inputs"], {"image": "1.png"})
            gen = first[prov._find_nodes(first)["generate"]]["inputs"]
            self.assertEqual(gen["first_frame"], ["1", 0])
            self.assertNotIn("last_frame", gen)
            self.assertNotIn("1", second)
            self.assertEqual(prov.client.uploads,
                             [os.path.join(image_dir, "1.png")])

    def test_no_match_falls_back_to_t2v(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.")
            image_dir = make_image_dir(tmp, "2.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            self.assertNotIn("1", prov.client.queued[-1])
            self.assertEqual(prov.client.uploads, [])

    def test_both_flags_initial_for_clip1_chain_for_clip2(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir,
                           chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            names = [w.get("1", {}).get("inputs", {}).get("image")
                     for w in prov.client.queued]
            # Clip 1: initial image; clip 2: previous clip last frame --
            # the initial image never overrides chaining.
            self.assertEqual(names, ["1.png", "scene_001_last.png"])

    def test_chain_unchanged_without_image_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_provider(), chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            names = [w.get("1", {}).get("inputs", {}).get("image")
                     for w in prov.client.queued]
            self.assertEqual(names, [None, "scene_001_last.png"])

    def test_resume_skips_valid_clip1_without_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)

            def fake_valid(path, *args):
                s = str(path)
                if s.endswith(".tmp"):
                    return True
                return "scene_001" in s

            run_stage(ctx, scenes, fake_valid, force_all=False)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            self.assertEqual(prov.client.uploads, [])

    def test_scene1_uses_image_when_generating(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, valid=True, only_scene=1)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            self.assertEqual(prov.client.queued[-1]["1"]["inputs"],
                             {"image": "1.png"})

    def test_scene2_without_chain_skips_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, valid=True, only_scene=2)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            self.assertNotIn("1", prov.client.queued[-1])
            self.assertEqual(prov.client.uploads, [])

    def test_manifest_durations_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            image_dir = make_image_dir(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(), image_1_dir=image_dir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            reqs = []
            orig = ctx["video_provider"].generate
            ctx["video_provider"].generate = (
                lambda r: (reqs.append(r), orig(r))[1])
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual([(r.duration, r.prompt) for r in reqs],
                             [(10, "One."), (10, "Two.")])
            self.assertEqual(reqs[0].first_frame, "1.png")
            self.assertIsNone(reqs[1].first_frame)

    def test_ltx_unchanged(self):
        from providers.video.ltx import LtxVideoProvider
        entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        ltx = LtxVideoProvider(name="ltx", settings=dict(entry),
                               client=StubClient())
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.")
            img = os.path.join(tmp, "frame.png")
            with open(img, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
            ctx = make_ctx(tmp, ltx)
            ctx["video_model"] = "ltx"
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            with patch.object(st, "valid_video_file", return_value=True), \
                 patch.object(st, "valid_image_file", return_value=True), \
                 patch.object(ffmod, "extract_last_frame",
                              side_effect=fake_extract):
                paths = main.run_video_stage(ctx, scenes, {1: img},
                                             force_all=True)
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(ctx["video_provider"].client.queued), 1)
            # Only the pre-existing LTX I2V reference upload; no
            # first-frame chaining machinery is involved.
            self.assertEqual(ctx["video_provider"].client.uploads, [img])


if __name__ == "__main__":
    unittest.main()
