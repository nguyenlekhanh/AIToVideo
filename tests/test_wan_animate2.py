"""Unit tests: Wan Animate2 motion-control backend. No GPU: ComfyUI mocked."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import state as st
from providers.comfy import ComfyClient
from providers.errors import ProviderError
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.wan_animate2 import WanAnimate2Provider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "video", "wan_animate2")
    settings = dict(entry, **overrides)
    return WanAnimate2Provider(name="wan_animate2", settings=settings,
                               client=ComfyClient("http://127.0.0.1:8188"))


def make_motion_dir(names=("001.mp4", "002.mp4", "010.mp4")):
    tmp = tempfile.TemporaryDirectory()
    for name in names:
        Path(tmp.name, name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return tmp


class WanRegistryTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("video", AI_VIDEO_DIR, "wan_animate2",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, WanAnimate2Provider)
        self.assertEqual(provider.provider_id, "wan_animate2")

    def test_ltx_unchanged(self):
        from providers.video.ltx import LtxVideoProvider
        provider = create_provider("video", AI_VIDEO_DIR, "ltx",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, LtxVideoProvider)
        self.assertFalse(getattr(provider, "uses_motion_clips", False))
        self.assertFalse(getattr(provider, "fixed_duration", False))

    def test_request_defaults(self):
        req = VideoRequest(prompt="p", negative_prompt="n", input_image=None,
                           width=1, height=1, duration=5.0, seed=1,
                           output_path=Path("x.mp4"))
        self.assertIsNone(req.motion_video)


class WanTemplateTest(unittest.TestCase):
    def test_loads_with_required_nodes(self):
        wf = make_provider().load_workflow()
        classes = {n.get("class_type") for n in wf.values()}
        for required in ("LoadImage", "LoadVideo", "WanAnimate2ToVideo",
                         "SamplerCustom", "CreateVideo", "SaveVideo",
                         "PrimitiveStringMultiline", "GetVideoComponents"):
            self.assertIn(required, classes)

    def test_find_nodes(self):
        found = make_provider()._find_nodes(make_provider().load_workflow())
        self.assertEqual(found["image"], "189")
        self.assertEqual(found["video"], "240")
        self.assertEqual(found["save"], "246")
        self.assertEqual(found["sampler"], "1019")
        self.assertEqual(found["char"], "605")
        self.assertEqual(found["pose"], "604")

    def test_missing_workflow_raises(self):
        provider = make_provider(
            workflow_path=os.path.join(AI_VIDEO_DIR, "nope.json"))
        with self.assertRaises(ProviderError):
            provider.load_workflow()


class WanCustomizeTest(unittest.TestCase):
    def test_injections(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        wf = provider.customize(template, reference_name="god.png",
                                motion_name="002.mp4",
                                character_prompt="A dance.", seed=11,
                                prefix="wan_animate2/002")
        self.assertEqual(template, before)  # template file content untouched
        self.assertEqual(wf["189"]["inputs"]["image"], "god.png")
        self.assertEqual(wf["240"]["inputs"]["file"], "002.mp4")
        self.assertEqual(wf["605"]["inputs"]["value"], "A dance.")
        self.assertEqual(wf["1019"]["inputs"]["noise_seed"], 11)
        self.assertEqual(wf["246"]["inputs"]["filename_prefix"],
                         "wan_animate2/002")

    def test_only_allowed_nodes_change(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(template, reference_name="r.png",
                                   motion_name="m.mp4", character_prompt="c",
                                   seed=1, prefix="p")
        allowed = {"LoadImage", "LoadVideo", "PrimitiveStringMultiline",
                   "SamplerCustom", "SaveVideo"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in {nid}")
        self.assertEqual(len([n for n in before if before[n] != after[n]]), 5)


class WanDiscoveryTest(unittest.TestCase):
    def test_natural_sort(self):
        with make_motion_dir(("010.mp4", "002.mp4", "001.mp4")) as tmp:
            found = WanAnimate2Provider.discover_motion_clips(tmp)
        self.assertEqual([os.path.basename(p) for p in found],
                         ["001.mp4", "002.mp4", "010.mp4"])

    def test_extensions_case_insensitive(self):
        with make_motion_dir(("a.MP4", "b.mov", "c.WEBM", "d.txt",
                              "e.gif")) as tmp:
            found = WanAnimate2Provider.discover_motion_clips(tmp)
        self.assertEqual(len(found), 3)

    def test_missing_dir(self):
        with self.assertRaises(ProviderError):
            WanAnimate2Provider.discover_motion_clips("/nope/motion")

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderError):
                WanAnimate2Provider.discover_motion_clips(tmp)


class WanGenerateTest(unittest.TestCase):
    class RecClient(ComfyClient):
        def __init__(self):
            super().__init__("http://127.0.0.1:8188")
            self.workflows = []
            self.uploads = []

        def upload_image(self, path):
            self.uploads.append(("image", path))
            return os.path.basename(path)

        def upload_video(self, path):
            self.uploads.append(("video", path))
            return os.path.basename(path)

        def run(self, wf, keys, dest):
            self.workflows.append(wf)
            Path(dest).write_bytes(b"fakevideo")
            return dest

    def _provider(self):
        entry = lookup(AI_VIDEO_DIR, "video", "wan_animate2")
        client = self.RecClient()
        provider = WanAnimate2Provider(name="wan_animate2",
                                       settings=dict(entry), client=client)
        return provider, client

    def test_reference_reused_across_jobs(self):
        provider, client = self._provider()
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "god.png")
            Path(ref).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
            for stem in ("001", "002", "003"):
                motion = os.path.join(tmp, f"{stem}.mp4")
                Path(motion).write_bytes(b"fake")
                provider.generate(VideoRequest(
                    prompt="dance", negative_prompt="", input_image=Path(ref),
                    width=482, height=854, duration=5.0, seed=1,
                    output_path=Path(tmp, f"out_{stem}.mp4"),
                    motion_video=Path(motion)))
        images = [w["189"]["inputs"]["image"] for w in client.workflows]
        self.assertEqual(images, ["god.png"] * 3)
        files = [w["240"]["inputs"]["file"] for w in client.workflows]
        self.assertEqual(files, ["001.mp4", "002.mp4", "003.mp4"])
        # reference uploaded once, each motion uploaded once
        ref_uploads = [u for u in client.uploads if u[0] == "image"]
        self.assertEqual(len(ref_uploads), 1)

    def test_missing_motion_fails_clearly(self):
        provider, _ = self._provider()
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "god.png")
            Path(ref).write_bytes(b"x")
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=Path(ref), width=1, height=1,
                               duration=5.0, seed=1,
                               output_path=Path(tmp, "o.mp4"))
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(req)
            self.assertIn("--motion-dir", str(ctx.exception))

    def test_missing_reference_fails_clearly(self):
        provider, _ = self._provider()
        with tempfile.TemporaryDirectory() as tmp:
            motion = os.path.join(tmp, "001.mp4")
            Path(motion).write_bytes(b"x")
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=Path(tmp, "missing.png"),
                               width=1, height=1, duration=5.0, seed=1,
                               output_path=Path(tmp, "o.mp4"),
                               motion_video=Path(motion))
            with self.assertRaises(ProviderError):
                provider.generate(req)


class WanStageTest(unittest.TestCase):
    class StubProvider:
        uses_motion_clips = True
        fixed_duration = True
        supports_pinned_reference = True

        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            Path(str(request.output_path)).write_bytes(b"fakevideo")
            return SimpleNamespace(path=Path(str(request.output_path)),
                                   width=482, height=854)

    def _ctx(self, tmp, clips, ref=None):
        state = st.new_state("w")
        if ref is None:
            ref = os.path.join(tmp, "god.png")
            Path(ref).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return {"project": "w", "base_dir": tmp, "base_seed": 1,
                "state": state, "state_path": os.path.join(tmp, "s.json"),
                "img_dir": tmp, "vid_dir": tmp, "video_provider": self.StubProvider(),
                "video_model": "wan_animate2", "vid_neg": "",
                "target_w": 482, "target_h": 854, "default_duration": 7.0,
                "reference_image": ref, "motion_clips": clips,
                "motion_dir": tmp}

    def _scenes(self, n=3):
        return [{"id": i + 1, "duration": 7, "image_prompt": "A.",
                 "video_prompt": "Move.", "narration": "Voice."}
                for i in range(n)]

    def test_scene_clip_mapping_and_ref(self):
        with make_motion_dir() as mdir:
            clips = WanAnimate2Provider.discover_motion_clips(mdir)
            with tempfile.TemporaryDirectory() as tmp:
                ref = os.path.join(tmp, "god.png")
                Path(ref).write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
                ctx = self._ctx(tmp, clips, ref)
                with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                     patch.object(st, "valid_image_file", return_value=True):
                    paths = main.run_video_stage(ctx, self._scenes(), {},
                                                 force_all=True)
                prov = ctx["video_provider"]
                self.assertEqual(len(prov.requests), 3)
                for i, req in enumerate(prov.requests):
                    self.assertEqual(str(req.input_image), ref)
                    self.assertEqual(req.motion_video,
                                     Path(clips[i]))
                    self.assertEqual(str(req.output_path),
                                     main.scene_path(tmp, i + 1, "mp4") + ".tmp")
                self.assertEqual(len(paths), 3)

    def test_scene_selection_processes_only_clip(self):
        with make_motion_dir() as mdir:
            clips = WanAnimate2Provider.discover_motion_clips(mdir)
            with tempfile.TemporaryDirectory() as tmp:
                ctx = self._ctx(tmp, clips)
                with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                     patch.object(st, "valid_image_file", return_value=True):
                    main.run_video_stage(ctx, self._scenes(), {}, only_scene=2)
                prov = ctx["video_provider"]
                self.assertEqual(len(prov.requests), 1)
                self.assertEqual(prov.requests[0].motion_video,
                                 Path(clips[1]))

    def test_resume_skips_valid_clips(self):
        with make_motion_dir(("001.mp4", "002.mp4")) as mdir:
            clips = WanAnimate2Provider.discover_motion_clips(mdir)
            with tempfile.TemporaryDirectory() as tmp:
                ctx = self._ctx(tmp, clips)
                with patch.object(st, "valid_video_file", return_value=True), \
                     patch.object(st, "valid_image_file", return_value=True):
                    paths = main.run_video_stage(
                        ctx, self._scenes(2), {}, force_all=False)
                self.assertEqual(len(ctx["video_provider"].requests), 0)
                self.assertEqual(len(paths), 2)

    def test_missing_clip_for_scene_fails(self):
        with make_motion_dir(("001.mp4",)) as mdir:
            clips = WanAnimate2Provider.discover_motion_clips(mdir)
            with tempfile.TemporaryDirectory() as tmp:
                ctx = self._ctx(tmp, clips)
                with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                     patch.object(st, "valid_image_file", return_value=True):
                    with self.assertRaises(ValueError) as err:
                        main.run_video_stage(ctx, self._scenes(2), {},
                                             force_all=True)
                self.assertIn("scene 2", str(err.exception))


class WanCliTest(unittest.TestCase):
    def test_flags_parse(self):
        args = main.parse_args(["--project", "b", "--resume", "--stage",
                                "video", "--video-model", "wan_animate2",
                                "--reference-image", "reference/god.png",
                                "--motion-dir", "motion"])
        self.assertEqual(args.video_model, "wan_animate2")
        self.assertEqual(args.reference_image, "reference/god.png")
        self.assertEqual(args.motion_dir, "motion")

    def test_planner_motion_flags(self):
        scenes = [{"id": 1, "duration": 7, "image_prompt": "A.",
                   "video_prompt": "M.", "narration": "V."}]
        with tempfile.TemporaryDirectory() as tmp:
            stages, problems = main.plan_resume_stages(
                scenes=scenes, img_dir=tmp, vid_dir=tmp, aud_dir=tmp,
                final_path=os.path.join(tmp, "f.mp4"), default_duration=7.0,
                stage="video", video_needs_images=False,
                check_video_duration=False)
            self.assertEqual(stages, ["video"])
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
