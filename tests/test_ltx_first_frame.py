"""Unit tests: LTX first-frame chaining + initial image (mocked, no GPU)."""
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
from providers.errors import ProviderError
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.ltx import LtxVideoProvider
from providers.video.ltx_t2v import LtxT2VVideoProvider

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


def make_t2v():
    entry = lookup(AI_VIDEO_DIR, "video", "ltx_t2v")
    return LtxT2VVideoProvider(name="ltx_t2v", settings=dict(entry),
                               client=StubClient())


def make_i2v():
    entry = lookup(AI_VIDEO_DIR, "video", "ltx")
    return LtxVideoProvider(name="ltx", settings=dict(entry),
                            client=StubClient())


def req(tmp, name="o.mp4", **over):
    base = dict(prompt="p", negative_prompt="", input_image=None,
                width=768, height=1344, duration=10, seed=1,
                output_path=Path(tmp) / name)
    base.update(over)
    return VideoRequest(**base)


def graph_classes(wf):
    return {n.get("class_type") for n in wf.values()}


class LtxT2VFirstFrameTest(unittest.TestCase):
    def test_supports_first_frame(self):
        self.assertTrue(make_t2v().supports_first_frame)

    def test_default_is_pure_t2v(self):
        prov = make_t2v()
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp))
            wf = prov.client.queued[-1]
        classes = graph_classes(wf)
        self.assertNotIn("LoadImage", classes)
        self.assertNotIn("LTXVPreprocess", classes)
        self.assertEqual(prov.client.uploads, [])

    def test_first_frame_routes_through_i2v_graph(self):
        prov = make_t2v()
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp, first_frame="scene_001_last.png"))
            wf = prov.client.queued[-1]
        # Real image conditioning: the verified I2V graph consumes it.
        classes = graph_classes(wf)
        self.assertIn("LoadImage", classes)
        self.assertIn("LTXVPreprocess", classes)
        self.assertIn("LTXVImgToVideoInplace", classes)
        loads = [n for n in wf.values() if n.get("class_type") == "LoadImage"]
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["inputs"]["image"], "scene_001_last.png")
        # No local upload: the pipeline already uploaded the frame.
        self.assertEqual(prov.client.uploads, [])

    def test_chained_duration_and_resolution_shared(self):
        prov = make_t2v()
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp, first_frame="f.png"))
            wf = prov.client.queued[-1]
        math = [n for n in wf.values()
                if n.get("class_type") == "ComfyMathExpression"
                and n.get("inputs", {}).get("expression") == "a * b + 1"]
        self.assertEqual(len(math), 1)
        self.assertEqual(len([n for n in wf.values()
                              if n.get("class_type") == "ResolutionSelector"]), 1)

    def test_missing_i2v_ref_fails_clearly(self):
        prov = make_t2v(workflow_path="x") if False else make_t2v()
        prov.settings = dict(prov.settings, workflow_ref_path="")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderError):
                prov.generate(req(tmp, first_frame="f.png"))


class LtxI2VFirstFrameTest(unittest.TestCase):
    def test_first_frame_uses_name_without_local_upload(self):
        prov = make_i2v()
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp, first_frame="1.png"))
            wf = prov.client.queued[-1]
        loads = [n for n in wf.values() if n.get("class_type") == "LoadImage"]
        self.assertEqual(loads[0]["inputs"]["image"], "1.png")
        self.assertEqual(prov.client.uploads, [])

    def test_input_image_path_still_uploads(self):
        prov = make_i2v()
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "ref.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
            prov.generate(req(tmp, input_image=img))
            wf = prov.client.queued[-1]
        loads = [n for n in wf.values() if n.get("class_type") == "LoadImage"]
        self.assertEqual(loads[0]["inputs"]["image"], "ref.png")
        self.assertEqual(prov.client.uploads, [str(img)])

    def test_imageless_falls_back_to_t2v_graph(self):
        prov = make_i2v()
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp))
            wf = prov.client.queued[-1]
        classes = graph_classes(wf)
        self.assertNotIn("LoadImage", classes)
        self.assertIn("PrimitiveStringMultiline", classes)
        self.assertEqual(prov.client.uploads, [])

    def test_registry_i2v_unchanged(self):
        prov = create_provider("video", AI_VIDEO_DIR, "ltx",
                               client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(prov, LtxVideoProvider)
        self.assertTrue(getattr(prov, "needs_input_image", True))


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


def make_ctx(tmp, provider, model, chain=False, image_1_dir=None):
    state = st.new_state("w")
    return {"project": "w", "base_dir": tmp, "base_seed": 1,
            "state": state, "state_path": os.path.join(tmp, "s.json"),
            "img_dir": tmp, "vid_dir": tmp, "video_provider": provider,
            "video_model": model, "vid_neg": "",
            "target_w": 768, "target_h": 1344, "default_duration": 5.0,
            "prompt_mode": True, "first_frame_chain": chain,
            "image_1_dir": image_1_dir, "ffmpeg_exe": "ffmpeg"}


def fake_extract(video_path, out_path, executable="ffmpeg"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"png")
    return out_path


def run_stage(ctx, scenes, valid=True, **kwargs):
    vf = (patch.object(st, "valid_video_file", return_value=valid)
          if isinstance(valid, bool)
          else patch.object(st, "valid_video_file", side_effect=valid))
    with vf, patch.object(st, "valid_image_file", return_value=False), \
         patch.object(ffmod, "extract_last_frame", side_effect=fake_extract):
        return main.run_video_stage(ctx, scenes, {}, **kwargs)


class LtxStageTest(unittest.TestCase):
    def _names(self, prov):
        out = []
        for wf in prov.client.queued:
            loads = [n for n in wf.values()
                     if n.get("class_type") == "LoadImage"]
            out.append(loads[0]["inputs"]["image"] if loads else None)
        return out

    def test_t2v_chain_priority_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.", "Three.")
            ctx = make_ctx(tmp, make_t2v(), "ltx_t2v", chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            # Clip 1 T2V; clips 2/3 chained via the I2V graph.
            self.assertEqual(self._names(ctx["video_provider"]),
                             [None, "scene_001_last.png", "scene_002_last.png"])

    def test_image1_only_no_chaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            imgdir = os.path.join(tmp, "image_start")
            os.makedirs(imgdir)
            with open(os.path.join(imgdir, "1.png"), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
            ctx = make_ctx(tmp, make_t2v(), "ltx_t2v", image_1_dir=imgdir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual(self._names(ctx["video_provider"]),
                             ["1.png", None])

    def test_both_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            imgdir = os.path.join(tmp, "image_start")
            os.makedirs(imgdir)
            with open(os.path.join(imgdir, "1.png"), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
            ctx = make_ctx(tmp, make_t2v(), "ltx_t2v", chain=True,
                           image_1_dir=imgdir)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual(self._names(ctx["video_provider"]),
                             ["1.png", "scene_001_last.png"])

    def test_i2v_key_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_i2v(), "ltx", chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual(self._names(ctx["video_provider"]),
                             [None, "scene_001_last.png"])

    def test_resume_skips_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.", "Three.")
            ctx = make_ctx(tmp, make_t2v(), "ltx_t2v", chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)

            def fake_valid(path, *args):
                s = str(path)
                if s.endswith(".tmp"):
                    return True
                return "scene_003" not in s

            run_stage(ctx, scenes, fake_valid, force_all=False)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.client.queued), 1)
            self.assertEqual(self._names(prov), ["scene_002_last.png"])

    def test_manifest_durations_reach_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, "One.", "Two.")
            ctx = make_ctx(tmp, make_t2v(), "ltx_t2v", chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            reqs = []
            orig = ctx["video_provider"].generate
            ctx["video_provider"].generate = (
                lambda r: (reqs.append(r), orig(r))[1])
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual([r.duration for r in reqs], [10, 10])
            self.assertEqual(reqs[1].first_frame, "scene_001_last.png")

    def test_minimax_unchanged(self):
        from providers.video.minimax_h3 import MiniMaxH3VideoProvider
        entry = lookup(AI_VIDEO_DIR, "video", "minimax_h3")
        prov = MiniMaxH3VideoProvider(name="minimax_h3",
                                      settings=dict(entry),
                                      client=StubClient())
        self.assertTrue(prov.supports_first_frame)
        with tempfile.TemporaryDirectory() as tmp:
            prov.generate(req(tmp, first_frame="f.png"))
            wf = prov.client.queued[-1]
        self.assertEqual(wf["1"]["inputs"], {"image": "f.png"})


class LtxCliTest(unittest.TestCase):
    def test_ltx_accepts_both_flags(self):
        args = main.parse_args(["--project", "m6", "--video-model", "ltx",
                                "--prompt-dir", "prompt", "--use-first-frame",
                                "--image-1-dir", "image_start",
                                "--aspect", "9:16", "--resolution", "720",
                                "--stop-after", "video"])
        self.assertIsNone(main.validate_cli_combination(args))

    def test_ltx_t2v_accepts_flags(self):
        args = main.parse_args(["--project", "m6", "--video-model", "ltx_t2v",
                                "--prompt-dir", "prompt", "--use-first-frame"])
        self.assertIsNone(main.validate_cli_combination(args))

    def test_wan_still_rejected(self):
        args = main.parse_args(["--project", "m6", "--video-model", "wan",
                                "--prompt-dir", "prompt", "--use-first-frame"])
        self.assertIsNotNone(main.validate_cli_combination(args))


if __name__ == "__main__":
    unittest.main()
