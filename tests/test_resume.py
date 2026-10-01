"""Unit + workflow tests for resume / per-scene regeneration (mocked providers).

No GPU generation: stub providers write fixture bytes (video stubs use the
local ffmpeg testsrc filter only where real durations matter). No network:
Ollama/ComfyUI are never touched (tests fail loudly if they are).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest import mock

import main
import ollama as ol
import storyboard as sb
import state as st
from providers.audio.base import AudioProvider, AudioRequest, AudioResult
from providers.errors import ProviderError
from providers.comfy import ComfyClient
from providers.image.base import ImageProvider, ImageRequest, ImageResult
from providers.video.base import VideoProvider, VideoRequest, VideoResult

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
MP3_BYTES = b"ID3\x04\x00" + b"\x00" * 200

DURATIONS = [5, 6, 7, 5, 8]


def fixture_scenes(durations=DURATIONS):
    scenes = []
    for i, duration in enumerate(durations, start=1):
        scenes.append({"id": i, "duration": duration,
                       "image_prompt": f"scene {i} image",
                       "video_prompt": f"scene {i} video",
                       "narration": f"Scene {i} narration without markers."})
    return scenes


def fixture_raw(durations=DURATIONS):
    return {"scenes": fixture_scenes(durations)}


class StubImageProvider(ImageProvider):
    provider_id = "stub-image"

    def __init__(self):
        self.calls = []
        self.fail_on = set()
        self.interrupt_on = set()
        self._nonce = 0

    def generate(self, request: ImageRequest) -> ImageResult:
        self.calls.append(request)
        sid = _scene_from_path(request.output_path)
        if sid in self.interrupt_on:
            raise KeyboardInterrupt()
        if sid in self.fail_on:
            raise ProviderError("stub-image", "stub image failure")
        path = Path(request.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._nonce += 1
        path.write_bytes(PNG_BYTES + bytes([self._nonce % 256]))
        return ImageResult(path=path, width=request.width, height=request.height)


class StubVideoProvider(VideoProvider):
    provider_id = "stub-video"

    def __init__(self):
        self.calls = []

    def generate(self, request: VideoRequest) -> VideoResult:
        self.calls.append(request)
        dest = Path(request.output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"testsrc=duration={float(request.duration)}:"
                   f"size=256x256:rate=10",
             "-pix_fmt", "yuv420p", "-c:v", "libx264",
             "-f", "mp4", str(dest)],
            check=True, capture_output=True)
        return VideoResult(path=dest, width=request.width, height=request.height)


class StubAudioProvider(AudioProvider):
    provider_id = "stub-audio"

    def __init__(self):
        self.calls = []

    def generate(self, request: AudioRequest) -> AudioResult:
        self.calls.append(request)
        path = Path(request.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MP3_BYTES)
        return AudioResult(path=path)


def _scene_from_path(path) -> int:
    match = re.search(r"scene_(\d+)", str(path))
    if not match:
        raise ValueError(f"Cannot parse scene id from {path}")
    return int(match.group(1))


def require_ffmpeg(testcase):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        testcase.skipTest("ffmpeg/ffprobe not available")


def make_ctx(tmpdir, stubs, durations=DURATIONS, seed=7):
    img_dir = os.path.join(tmpdir, "images")
    vid_dir = os.path.join(tmpdir, "videos")
    aud_dir = os.path.join(tmpdir, "audio")
    mux_dir = os.path.join(tmpdir, "muxed")
    for d in (img_dir, vid_dir, aud_dir, mux_dir):
        os.makedirs(d, exist_ok=True)
    state_path = os.path.join(tmpdir, "state.json")
    state = st.new_state("test")
    state["scene_ids"] = list(range(1, len(durations) + 1))
    state["base_seed"] = seed
    return {
        "project": "test", "img_dir": img_dir, "vid_dir": vid_dir,
        "aud_dir": aud_dir, "mux_dir": mux_dir,
        "final_path": os.path.join(tmpdir, "final.mp4"),
        "storyboard_path": os.path.join(tmpdir, "storyboard.json"),
        "base_dir": tmpdir, "state_path": state_path, "state": state,
        "image_provider": stubs["image"], "video_provider": stubs["video"],
        "audio_provider": stubs["audio"],
        "image_model": "stub", "video_model": "stub",
        "target_w": 512, "target_h": 512, "base_seed": seed,
        "img_neg": "", "vid_neg": "", "voice": "v",
        "rate": "+0%", "pitch": "+0Hz", "default_duration": 5.0,
        "char_ref": None, "subject": None, "ffmpeg_exe": "ffmpeg",
    }


def write_storyboard(tmpdir, durations=DURATIONS):
    sb.save_storyboard(fixture_scenes(durations),
                       os.path.join(tmpdir, "storyboard.json"))


class PlannerTest(unittest.TestCase):
    def test_continue_after_partial_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx_dirs = make_ctx(tmp, {"image": None, "video": None, "audio": None})
            for sid in (1, 2, 4):
                Path(main.scene_path(ctx_dirs["img_dir"], sid, "png")).write_bytes(PNG_BYTES)
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes([5, 6, 7, 5, 8]),
                img_dir=ctx_dirs["img_dir"], vid_dir=ctx_dirs["vid_dir"],
                aud_dir=ctx_dirs["aud_dir"],
                final_path=ctx_dirs["final_path"], default_duration=5.0)
            self.assertEqual(problems, [])
            self.assertEqual(stages[0], "image")

    def test_all_complete_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, {"image": None, "video": None, "audio": None})
            for sid in range(1, 6):
                Path(main.scene_path(ctx["img_dir"], sid, "png")).write_bytes(PNG_BYTES)
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=ctx["img_dir"],
                vid_dir=ctx["vid_dir"], aud_dir=ctx["aud_dir"],
                final_path=ctx["final_path"], default_duration=5.0)
            # videos missing -> first incomplete is video
            self.assertEqual(stages[0], "video")
            self.assertEqual(problems, [])

    def test_explicit_stage_video_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, {"image": None, "video": None, "audio": None})
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=ctx["img_dir"],
                vid_dir=ctx["vid_dir"], aud_dir=ctx["aud_dir"],
                final_path=ctx["final_path"], default_duration=5.0,
                stage="video", only_scene=3)
            self.assertEqual(stages, [])
            self.assertTrue(any("scene_003" in p for p in problems))

    def test_explicit_stage_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(tmp, {"image": None, "video": None, "audio": None})
            Path(main.scene_path(ctx["img_dir"], 3, "png")).write_bytes(PNG_BYTES)
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=ctx["img_dir"],
                vid_dir=ctx["vid_dir"], aud_dir=ctx["aud_dir"],
                final_path=ctx["final_path"], default_duration=5.0,
                stage="video", only_scene=3)
            self.assertEqual(stages, ["video"])
            self.assertEqual(problems, [])


class ImageStageTest(unittest.TestCase):
    def test_generates_missing_only_and_skips_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            pre = main.scene_path(ctx["img_dir"], 2, "png")
            Path(pre).write_bytes(PNG_BYTES)
            paths = main.run_image_stage(ctx, fixture_scenes())
            self.assertEqual(sorted(paths), [1, 2, 3, 4, 5])
            generated = sorted(_scene_from_path(r.output_path)
                               for r in stubs["image"].calls)
            self.assertEqual(generated, [1, 3, 4, 5])

    def test_scene_forces_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            before = {}
            for sid in range(1, 6):
                dest = main.scene_path(ctx["img_dir"], sid, "png")
                Path(dest).write_bytes(PNG_BYTES + bytes([sid]))
                before[sid] = Path(dest).read_bytes()
            main.run_image_stage(ctx, fixture_scenes(), only_scene=3)
            generated = [_scene_from_path(r.output_path)
                         for r in stubs["image"].calls]
            self.assertEqual(generated, [3])
            for sid in range(1, 6):
                content = Path(main.scene_path(ctx["img_dir"], sid, "png")).read_bytes()
                if sid == 3:
                    self.assertNotEqual(content, before[sid])
                else:
                    self.assertEqual(content, before[sid])

    def test_corrupt_output_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            dest = main.scene_path(ctx["img_dir"], 1, "png")
            Path(dest).write_bytes(b"not an image")
            main.run_image_stage(ctx, fixture_scenes([5]), only_scene=None)
            self.assertTrue(st.valid_image_file(dest))

    def test_failure_preserves_good_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            dest = main.scene_path(ctx["img_dir"], 1, "png")
            Path(dest).write_bytes(PNG_BYTES)
            stubs["image"].fail_on = {1}
            with self.assertRaises(ProviderError):
                main.run_image_stage(ctx, fixture_scenes([5]), only_scene=1)
            self.assertEqual(Path(dest).read_bytes(), PNG_BYTES)
            self.assertFalse(os.path.isfile(dest + ".tmp"))

    def test_ctrl_c_safe_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            stubs["image"].interrupt_on = {4}
            with self.assertRaises(main.StageInterrupted) as ctx_err:
                main.run_image_stage(ctx, fixture_scenes())
            message = ctx_err.exception.message
            self.assertIn("scene 4", message)
            self.assertIn("--resume", message)
            for sid in (1, 2, 3):
                self.assertTrue(st.valid_image_file(
                    main.scene_path(ctx["img_dir"], sid, "png")))
            self.assertFalse(os.path.isfile(
                main.scene_path(ctx["img_dir"], 4, "png")))
            completed = st.completed_scenes(ctx["state"], "image")
            self.assertEqual(completed, [1, 2, 3])
            # state file itself is valid JSON on disk
            reloaded = st.load_state(ctx["state_path"], "test")
            self.assertEqual(st.completed_scenes(reloaded, "image"), [1, 2, 3])

    def test_no_network_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": StubImageProvider(), "video": None, "audio": None}
            ctx = make_ctx(tmp, stubs)
            with mock.patch.object(ol, "generate_storyboard",
                                   side_effect=AssertionError("no ollama")):
                with mock.patch.object(sb, "save_storyboard",
                                       side_effect=AssertionError("no save")):
                    main.run_image_stage(ctx, fixture_scenes())


class VideoStageTest(unittest.TestCase):
    def test_durations_preserved_stub(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": None, "video": StubVideoProvider(), "audio": None}
            ctx = make_ctx(tmp, stubs)
            images = {}
            for sid in range(1, 6):
                dest = main.scene_path(ctx["img_dir"], sid, "png")
                Path(dest).write_bytes(PNG_BYTES)
                images[sid] = dest
            main.run_video_stage(ctx, fixture_scenes(), images)
            durations = [r.duration for r in stubs["video"].calls]
            self.assertEqual(durations, [5.0, 6.0, 7.0, 5.0, 8.0])
            self.assertEqual(durations[1], 6.0)
            self.assertNotEqual(durations[2], 5.0)

    def test_seven_seconds_not_five(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": None, "video": StubVideoProvider(), "audio": None}
            ctx = make_ctx(tmp, stubs)
            dest = main.scene_path(ctx["img_dir"], 1, "png")
            Path(dest).write_bytes(PNG_BYTES)
            main.run_video_stage(
                ctx, fixture_scenes([7]), {1: dest}, only_scene=1)
            self.assertEqual(stubs["video"].calls[0].duration, 7.0)

    def test_missing_image_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": None, "video": StubVideoProvider(), "audio": None}
            ctx = make_ctx(tmp, stubs)
            with self.assertRaises(ValueError) as ctx_err:
                main.run_video_stage(ctx, fixture_scenes([5]), {}, only_scene=1)
            self.assertIn("scene 1", str(ctx_err.exception))
            self.assertIn("Generate images first", str(ctx_err.exception))
            self.assertEqual(stubs["video"].calls, [])

    def test_skips_valid_clips(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": None, "video": StubVideoProvider(), "audio": None}
            ctx = make_ctx(tmp, stubs)
            images = {}
            for sid, duration in zip(range(1, 6), DURATIONS):
                img = main.scene_path(ctx["img_dir"], sid, "png")
                Path(img).write_bytes(PNG_BYTES)
                images[sid] = img
            main.run_video_stage(ctx, fixture_scenes(), images)
            first_calls = len(stubs["video"].calls)
            self.assertEqual(first_calls, 5)
            main.run_video_stage(ctx, fixture_scenes(), images)
            self.assertEqual(len(stubs["video"].calls), first_calls)


class AudioStageTest(unittest.TestCase):
    def test_generates_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            stubs = {"image": None, "video": None, "audio": StubAudioProvider()}
            ctx = make_ctx(tmp, stubs)
            paths = main.run_audio_stage(ctx, fixture_scenes())
            self.assertEqual(sorted(paths), [1, 2, 3, 4, 5])
            for sid in range(1, 6):
                self.assertTrue(st.valid_audio_file(
                    main.scene_path(ctx["aud_dir"], sid, "mp3")))
            calls_before = len(stubs["audio"].calls)
            main.run_audio_stage(ctx, fixture_scenes())
            self.assertEqual(len(stubs["audio"].calls), calls_before)


class ResumeIntegrationTest(unittest.TestCase):
    """Task Phase 7: storyboard -> interrupt -> resume -> regen -> video."""

    def test_full_resume_flow(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            img_stub = StubImageProvider()
            vid_stub = StubVideoProvider()
            stubs = {"image": img_stub, "video": vid_stub, "audio": StubAudioProvider()}
            ctx = make_ctx(tmp, stubs)
            write_storyboard(tmp)
            scenes = main.load_project_scenes(tmp)
            self.assertEqual(len(scenes), 5)
            with mock.patch.object(ol, "generate_storyboard",
                                   side_effect=AssertionError("no ollama")), \
                 mock.patch.object(sb, "save_storyboard",
                                   side_effect=AssertionError("no save")):
                # 1-2: images complete for 1..3, then interruption at 4.
                img_stub.interrupt_on = {4}
                with self.assertRaises(main.StageInterrupted):
                    main.run_image_stage(ctx, scenes)
                self.assertEqual(st.completed_scenes(ctx["state"], "image"), [1, 2, 3])
                img_stub.interrupt_on = set()
                # 3-4: resume generates the interrupted scene 4, then 5.
                n_before = len(img_stub.calls)
                main.run_image_stage(ctx, scenes)
                generated = sorted(_scene_from_path(r.output_path)
                                   for r in img_stub.calls[n_before:])
                self.assertEqual(generated, [4, 5])
                # 5: explicit regen of scene 3 only.
                before = {sid: Path(main.scene_path(ctx["img_dir"], sid, "png")).read_bytes()
                          for sid in range(1, 6)}
                n_calls = len(img_stub.calls)
                main.run_image_stage(ctx, scenes, only_scene=3)
                self.assertEqual(len(img_stub.calls), n_calls + 1)
                for sid in range(1, 6):
                    content = Path(main.scene_path(ctx["img_dir"], sid, "png")).read_bytes()
                    self.assertEqual(content != before[sid], sid == 3)
                # 6-7: video stage runs with storyboard durations.
                images = {sid: main.scene_path(ctx["img_dir"], sid, "png")
                          for sid in range(1, 6)}
                main.run_video_stage(ctx, scenes, images)
                self.assertEqual([r.duration for r in vid_stub.calls],
                                 [5.0, 6.0, 7.0, 5.0, 8.0])
            # storyboard file untouched throughout.
            with open(os.path.join(tmp, "storyboard.json"), encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(len(raw["scenes"]), 5)


class MainCommandsTest(unittest.TestCase):
    """COMMAND A/B/C through main.main with mocked providers/transports."""

    PROJECT = "resume_cmd_tmp"

    def setUp(self):
        require_ffmpeg(self)
        self.stubs = {"image": StubImageProvider(), "video": StubVideoProvider(),
                      "audio": StubAudioProvider()}
        base = os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT)
        shutil.rmtree(base, ignore_errors=True)
        self.addCleanup(shutil.rmtree, base, True)
        patches = [
            mock.patch.object(main, "create_provider",
                              side_effect=lambda kind, *_a, **_k: self.stubs[kind]),
            mock.patch.object(ol, "check_reachable", return_value=None),
            mock.patch.object(ComfyClient, "check_reachable", return_value=None),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _base(self):
        return os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT)

    def test_command_a_stops_after_images(self):
        with mock.patch.object(
                ol, "generate_storyboard",
                return_value={"scenes": fixture_scenes()}) as storyboard_mock:
            code = main.main(["test prompt", "--project", self.PROJECT,
                              "--research", "none", "--image-model", "sd15",
                              "--video-model", "ltx", "--aspect", "16:9",
                              "--resolution", "720", "--scenes", "5",
                              "--stop-after", "image"])
        self.assertEqual(code, 0)
        base = self._base()
        with open(os.path.join(base, "storyboard.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(len(raw["scenes"]), 5)
        for sid in range(1, 6):
            self.assertTrue(st.valid_image_file(
                os.path.join(base, "images", f"scene_{sid:03d}.png")))
        self.assertEqual(
            [f for f in os.listdir(os.path.join(base, "videos"))
             if f.endswith(".mp4")], [])
        self.assertEqual(len(self.stubs["image"].calls), 5)
        self.assertEqual(self.stubs["video"].calls, [])
        self.assertEqual(storyboard_mock.call_count, 1)

    def test_command_b_regenerates_only_scene_3(self):
        self.test_command_a_stops_after_images()
        base = self._base()
        before = {}
        for sid in range(1, 6):
            path = os.path.join(base, "images", f"scene_{sid:03d}.png")
            before[sid] = Path(path).read_bytes()
        storyboard_before = Path(
            os.path.join(base, "storyboard.json")).read_bytes()
        self.stubs["image"].calls.clear()
        with mock.patch.object(ol, "generate_storyboard",
                               side_effect=AssertionError("no ollama")), \
             mock.patch.object(sb, "save_storyboard",
                               side_effect=AssertionError("no save")):
            code = main.main(["--project", self.PROJECT, "--resume",
                              "--stage", "image", "--scene", "3",
                              "--image-model", "krea"])
        self.assertEqual(code, 0)
        generated = [_scene_from_path(r.output_path)
                     for r in self.stubs["image"].calls]
        self.assertEqual(generated, [3])
        for sid in range(1, 6):
            content = Path(os.path.join(
                base, "images", f"scene_{sid:03d}.png")).read_bytes()
            self.assertEqual(content != before[sid], sid == 3)
        self.assertEqual(Path(os.path.join(base, "storyboard.json")).read_bytes(),
                         storyboard_before)

    def test_command_c_video_only_with_durations(self):
        self.test_command_a_stops_after_images()
        self.stubs["image"].calls.clear()
        with mock.patch.object(ol, "generate_storyboard",
                               side_effect=AssertionError("no ollama")), \
             mock.patch.object(sb, "save_storyboard",
                               side_effect=AssertionError("no save")):
            code = main.main(["--project", self.PROJECT, "--resume",
                              "--stage", "video", "--video-model", "ltx"])
        self.assertEqual(code, 0)
        self.assertEqual(self.stubs["image"].calls, [])
        durations = [r.duration for r in self.stubs["video"].calls]
        self.assertEqual(durations, [5.0, 6.0, 7.0, 5.0, 8.0])
        base = self._base()
        for sid, expected in zip(range(1, 6), durations):
            path = os.path.join(base, "videos", f"scene_{sid:03d}.mp4")
            self.assertTrue(st.valid_video_file(path, expected),
                            f"{path} should be a valid {expected}s clip")


def require_ffmpeg(testcase):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        testcase.skipTest("ffmpeg/ffprobe not available")


if __name__ == "__main__":
    unittest.main()
