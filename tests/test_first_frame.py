"""Unit tests: optional MiniMax first-frame chaining (mocked, no GPU)."""
from __future__ import annotations

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
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.minimax_h3 import MiniMaxH3VideoProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubClient(ComfyClient):
    """Records workflows/uploads; no server, no files except dest stubs."""

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
    return d


def make_ctx(tmp, provider, chain=True):
    state = st.new_state("w")
    return {"project": "w", "base_dir": tmp, "base_seed": 1,
            "state": state, "state_path": os.path.join(tmp, "s.json"),
            "img_dir": tmp, "vid_dir": tmp, "video_provider": provider,
            "video_model": "minimax_h3", "vid_neg": "",
            "target_w": 480, "target_h": 864, "default_duration": 10.0,
            "prompt_mode": True, "first_frame_chain": chain,
            "ffmpeg_exe": "ffmpeg"}


def fake_extract(video_path, out_path, executable="ffmpeg"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"png")
    return out_path


def run_stage(ctx, scenes, **kwargs):
    with patch.object(st, "valid_image_file", return_value=False), \
         patch.object(ffmod, "extract_last_frame", side_effect=fake_extract):
        return main.run_video_stage(ctx, scenes, {}, **kwargs)


class CliTest(unittest.TestCase):
    def test_flag_defaults_disabled(self):
        args = main.parse_args(["--project", "w", "--prompt-dir", "prompt"])
        self.assertFalse(args.use_first_frame)

    def test_valid_combination(self):
        args = main.parse_args(["--project", "w", "--prompt-dir", "prompt",
                                "--video-model", "minimax_h3",
                                "--use-first-frame"])
        self.assertIsNone(main.validate_cli_combination(args))

    def test_requires_prompt_dir(self):
        args = main.parse_args(["--project", "w", "--video-model",
                                "minimax_h3", "--use-first-frame"])
        self.assertIn("--prompt-dir", main.validate_cli_combination(args))

    def test_rejects_non_minimax(self):
        for model in ("ltx", "wan", "wan_animate2"):
            args = main.parse_args(["--project", "w", "--prompt-dir", "prompt",
                                    "--video-model", model,
                                    "--use-first-frame"])
            err = main.validate_cli_combination(args)
            self.assertIn("minimax_h3", err)


class ProviderChainTest(unittest.TestCase):
    def test_default_generate_has_no_frame_nodes(self):
        prov = make_provider()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=None, width=480, height=864,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            prov.generate(req)
            wf = prov.client.queued[-1]
        self.assertNotIn("1", wf)
        gen = wf[prov._find_nodes(wf)["generate"]]["inputs"]
        self.assertNotIn("first_frame", gen)
        self.assertNotIn("last_frame", gen)
        self.assertEqual(prov.client.uploads, [])

    def test_first_frame_mutates_correct_node(self):
        prov = make_provider()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=None, width=480, height=864,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4",
                               first_frame="scene_001_last.png")
            prov.generate(req)
            wf = prov.client.queued[-1]
        self.assertEqual(wf["1"],
                         {"class_type": "LoadImage",
                          "inputs": {"image": "scene_001_last.png"}})
        gen = wf[prov._find_nodes(wf)["generate"]]["inputs"]
        self.assertEqual(gen["first_frame"], ["1", 0])
        self.assertNotIn("last_frame", gen)

    def test_template_file_unchanged(self):
        prov = make_provider()
        wf = prov.load_workflow()
        classes = {n.get("class_type") for n in wf.values()}
        self.assertNotIn("LoadImage", classes)


class StageChainTest(unittest.TestCase):
    def test_clip1_pure_t2v(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)
            with patch.object(st, "valid_video_file", return_value=True):
                run_stage(ctx, [scenes[0]], force_all=True)
            prov = ctx["video_provider"]
            wf = prov.client.queued[-1]
            self.assertNotIn("1", wf)
            self.assertEqual(prov.client.uploads, [])

    def test_clip2_uses_clip1_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)
            with patch.object(st, "valid_video_file", return_value=True):
                run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 2)
            first, second = prov.client.queued
            self.assertNotIn("1", first)
            self.assertEqual(second["1"]["inputs"],
                             {"image": "scene_001_last.png"})
            gen = second[prov._find_nodes(second)["generate"]]["inputs"]
            self.assertEqual(gen["first_frame"], ["1", 0])
            self.assertNotIn("last_frame", gen)
            frame = os.path.join(tmp, "continuity", "scene_001_last.png")
            self.assertEqual(prov.client.uploads, [frame])

    def test_clip3_chains_from_clip2(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.", "Three.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)
            with patch.object(st, "valid_video_file", return_value=True):
                run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            names = [w.get("1", {}).get("inputs", {}).get("image")
                     for w in prov.client.queued]
            self.assertEqual(names, [None, "scene_001_last.png",
                                     "scene_002_last.png"])

    def test_resume_skips_valid_uses_prev_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.", "Three.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)

            def fake_valid(path, *args):
                s = str(path)
                if s.endswith(".tmp"):
                    return True
                return "scene_003" not in s

            with patch.object(st, "valid_video_file", side_effect=fake_valid):
                paths = run_stage(ctx, scenes, force_all=False)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            wf = prov.client.queued[-1]
            self.assertEqual(wf["1"]["inputs"],
                             {"image": "scene_002_last.png"})
            self.assertEqual(len(paths), 3)

    def test_missing_previous_clip_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)

            def fake_valid(path, *args):
                return str(path).endswith(".tmp")

            with patch.object(st, "valid_video_file", side_effect=fake_valid):
                with self.assertRaises(ValueError) as err:
                    run_stage(ctx, scenes, only_scene=2)
            self.assertIn("scene_001.mp4", str(err.exception))
            self.assertEqual(len(ctx["video_provider"].client.queued), 0)

    def test_scene3_requires_scene2(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.", "Three.")
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)

            def fake_valid(path, *args):
                s = str(path)
                if s.endswith(".tmp"):
                    return True
                return "scene_002" in s

            with patch.object(st, "valid_video_file", side_effect=fake_valid):
                run_stage(ctx, scenes, only_scene=3)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            self.assertEqual(prov.client.queued[-1]["1"]["inputs"],
                             {"image": "scene_002_last.png"})

    def test_manifest_durations_preserved(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            with open(os.path.join(prompt_dir, "manifest.json"), "w",
                       encoding="utf-8") as f:
                json.dump({"clips": {"1": {"duration": 10},
                                     "2": {"duration": 10}}}, f)
            ctx = make_ctx(tmp, make_provider())
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            reqs = []
            orig = ctx["video_provider"].generate
            ctx["video_provider"].generate = (
                lambda r: (reqs.append(r), orig(r))[1])
            with patch.object(st, "valid_video_file", return_value=True):
                run_stage(ctx, scenes, force_all=True)
            self.assertEqual([r.duration for r in reqs], [10, 10])
            self.assertEqual(reqs[1].first_frame, "scene_001_last.png")

    def test_chain_disabled_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_provider(), chain=False)
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)
            with patch.object(st, "valid_video_file", return_value=True):
                run_stage(ctx, scenes, force_all=True)
            prov = ctx["video_provider"]
            for wf in prov.client.queued:
                self.assertNotIn("1", wf)
            self.assertEqual(prov.client.uploads, [])

    def test_unsupported_backend_rejected_in_stage(self):
        from providers.video.ltx import LtxVideoProvider
        entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        ltx = LtxVideoProvider(name="ltx", settings=dict(entry),
                               client=StubClient())
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.")
            ctx = make_ctx(tmp, ltx)
            ctx["video_model"] = "ltx"
            scenes = pf.load_prompt_scenes(prompt_dir, 10.0)
            with patch.object(st, "valid_video_file",
                              side_effect=lambda p, *a: str(p).endswith(".tmp")):
                with self.assertRaises(ValueError) as err:
                    run_stage(ctx, scenes, force_all=True)
            self.assertIn("--use-first-frame", str(err.exception))


if __name__ == "__main__":
    unittest.main()
