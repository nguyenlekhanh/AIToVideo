"""Unit + workflow tests for --stage finalize (mocked media, real ffmpeg).

No GPU generation: scene clips/narration are tiny ffmpeg-generated fixtures
(skipped when ffmpeg/ffprobe are unavailable). No network: Ollama/ComfyUI
are never touched (tests fail loudly if they are).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ffmpeg as ff
import main
import ollama as ol
import storyboard as sb
import state as st
from providers.comfy import ComfyClient

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATIONS = [5, 6, 7]


def require_ffmpeg(testcase):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        testcase.skipTest("ffmpeg/ffprobe not available")


def fixture_scenes(durations=DURATIONS):
    scenes = []
    for i, duration in enumerate(durations, start=1):
        scenes.append({"id": i, "duration": duration,
                       "image_prompt": f"scene {i} image",
                       "video_prompt": f"scene {i} video",
                       "narration": f"Scene {i} narration without markers."})
    return scenes


def write_clip(path, seconds):
    # NOTE: clips carry a competing stereo audio track, exactly like real
    # LTX outputs. Without explicit -map, ffmpeg auto-selection prefers this
    # stereo track over the mono narration and silently drops the narration.
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=256x256:rate=10",
         "-f", "lavfi",
         "-i", f"sine=frequency=880:duration={seconds}",
         "-map", "0:v:0", "-map", "1:a:0",
         "-pix_fmt", "yuv420p", "-c:v", "libx264",
         "-c:a", "aac", "-ac", "2", "-ar", "48000",
         "-f", "mp4", path],
        check=True, capture_output=True)


def audio_channels(path):
    """Channel count of the first audio stream (proves which track won)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    return int(r.stdout.strip())


def write_audio(path, seconds):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", f"sine=frequency=440:duration={seconds}",
             "-c:a", "libmp3lame", path],
            check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise unittest.SkipTest(f"mp3 encoder unavailable: {exc}")


def make_project(tmpdir, durations=DURATIONS):
    """Storyboard + matching valid clips/narration. Returns paths dict."""
    sb.save_storyboard(fixture_scenes(durations),
                       os.path.join(tmpdir, "storyboard.json"))
    for sub in ("videos", "audio", "muxed"):
        os.makedirs(os.path.join(tmpdir, sub), exist_ok=True)
    for sid, seconds in enumerate(durations, start=1):
        write_clip(os.path.join(tmpdir, "videos", f"scene_{sid:03d}.mp4"), seconds)
        write_audio(os.path.join(tmpdir, "audio", f"scene_{sid:03d}.mp3"), seconds)
    return {
        "storyboard": os.path.join(tmpdir, "storyboard.json"),
        "vid_dir": os.path.join(tmpdir, "videos"),
        "aud_dir": os.path.join(tmpdir, "audio"),
        "mux_dir": os.path.join(tmpdir, "muxed"),
        "final": os.path.join(tmpdir, "final.mp4"),
        "state": os.path.join(tmpdir, "state.json"),
    }


def make_ctx(tmpdir, paths):
    state = st.new_state("test")
    state["scene_ids"] = [1, 2, 3]
    return {
        "project": "test", "mux_dir": paths["mux_dir"],
        "final_path": paths["final"], "state_path": paths["state"],
        "state": state, "ffmpeg_exe": "ffmpeg",
    }


class PlannerFinalizeTest(unittest.TestCase):
    def test_finalize_ok(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=os.path.join(tmp, "images"),
                vid_dir=paths["vid_dir"], aud_dir=paths["aud_dir"],
                final_path=paths["final"], default_duration=5.0,
                stage="finalize")
            self.assertEqual(stages, ["finalize"])
            self.assertEqual(problems, [])

    def test_finalize_missing_clip(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            os.unlink(os.path.join(paths["vid_dir"], "scene_002.mp4"))
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=os.path.join(tmp, "images"),
                vid_dir=paths["vid_dir"], aud_dir=paths["aud_dir"],
                final_path=paths["final"], default_duration=5.0,
                stage="finalize")
            self.assertEqual(stages, [])
            self.assertTrue(any("scene_002" in p for p in problems))
            self.assertTrue(any("--stage video" in p for p in problems))

    def test_finalize_missing_audio(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            os.unlink(os.path.join(paths["aud_dir"], "scene_003.mp3"))
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=os.path.join(tmp, "images"),
                vid_dir=paths["vid_dir"], aud_dir=paths["aud_dir"],
                final_path=paths["final"], default_duration=5.0,
                stage="finalize")
            self.assertEqual(stages, [])
            self.assertTrue(any("scene_003" in p for p in problems))
            self.assertTrue(any("--stage audio" in p for p in problems))

    def test_finalize_invalid_media(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            with open(os.path.join(paths["vid_dir"], "scene_001.mp4"), "wb") as f:
                f.write(b"not a video file")
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=os.path.join(tmp, "images"),
                vid_dir=paths["vid_dir"], aud_dir=paths["aud_dir"],
                final_path=paths["final"], default_duration=5.0,
                stage="finalize")
            self.assertEqual(stages, [])
            self.assertTrue(any("scene_001" in p for p in problems))

    def test_finalize_rejects_scene_filter(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            stages, problems = main.plan_resume_stages(
                scenes=fixture_scenes(), img_dir=os.path.join(tmp, "images"),
                vid_dir=paths["vid_dir"], aud_dir=paths["aud_dir"],
                final_path=paths["final"], default_duration=5.0,
                stage="finalize", only_scene=2)
            self.assertEqual(stages, [])
            self.assertTrue(any("--scene" in p for p in problems))


class FinalizeStageTest(unittest.TestCase):
    def test_mux_selects_narration_over_clip_audio(self):
        # Core regression: the clip's embedded stereo track must NOT win
        # over the mono narration (ffmpeg auto-selection picks most
        # channels). mux_scene maps 0:v:0 + 1:a:0 explicitly.
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            clip = os.path.join(tmp, "clip.mp4")
            audio = os.path.join(tmp, "narr.mp3")
            write_clip(clip, 3)
            write_audio(audio, 3)
            self.assertEqual(audio_channels(clip), 2)  # competing track present
            self.assertEqual(audio_channels(audio), 1)  # narration is mono
            out = os.path.join(tmp, "muxed.mp4")
            ff.mux_scene(clip, audio, out, copy_video=True)
            self.assertEqual(ff.probe_streams(out), ["video", "audio"])
            self.assertEqual(audio_channels(out), 1)  # narration won
            self.assertAlmostEqual(ff.probe_duration(out), 3.0, delta=0.5)

    def test_order_and_pairing(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            ctx = make_ctx(tmp, paths)
            scenes = fixture_scenes()
            pairs, concats = [], []
            real_mux, real_concat = ff.mux_scene, ff.concat_scenes

            def spy_mux(video, audio, out, **kwargs):
                pairs.append((os.path.basename(video), os.path.basename(audio)))
                return real_mux(video, audio, out, **kwargs)

            def spy_concat(scene_paths, out, **kwargs):
                concats.append([os.path.basename(p) for p in scene_paths])
                return real_concat(scene_paths, out, **kwargs)

            with mock.patch.object(ff, "mux_scene", side_effect=spy_mux), \
                 mock.patch.object(ff, "concat_scenes", side_effect=spy_concat):
                clips = {sid: os.path.join(paths["vid_dir"], f"scene_{sid:03d}.mp4")
                         for sid in (1, 2, 3)}
                audios = {sid: os.path.join(paths["aud_dir"], f"scene_{sid:03d}.mp3")
                          for sid in (1, 2, 3)}
                result = main.run_finalize_stage(ctx, scenes, clips, audios)
            self.assertEqual(
                pairs,
                [(f"scene_{i:03d}.mp4", f"scene_{i:03d}.mp3") for i in (1, 2, 3)])
            self.assertEqual(len(concats), 1)
            self.assertEqual(
                [os.path.splitext(p)[0] for p in concats[0]],
                ["scene_001", "scene_002", "scene_003"])
            self.assertEqual(result, paths["final"])
            self.assertTrue(st.valid_final_output(paths["final"]))
            final_dur = ff.probe_duration(paths["final"])
            self.assertAlmostEqual(final_dur, 5 + 6 + 7, delta=1.0)
            # Every muxed scene carries the narration (mono), not the
            # clip's embedded stereo track.
            for sid in (1, 2, 3):
                muxed = os.path.join(paths["mux_dir"], f"scene_{sid:03d}.mp4")
                self.assertEqual(audio_channels(muxed), 1)
            # No .tmp.mp4 leftovers.
            leftovers = [f for f in os.listdir(paths["mux_dir"]) if ".tmp." in f]
            self.assertEqual(leftovers, [])
            leftover_final = [f for f in os.listdir(tmp) if ".tmp." in f]
            self.assertEqual(leftover_final, [])

    def test_missing_clip_fails_without_output(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            os.unlink(os.path.join(paths["vid_dir"], "scene_002.mp4"))
            ctx = make_ctx(tmp, paths)
            clips = {sid: os.path.join(paths["vid_dir"], f"scene_{sid:03d}.mp4")
                     for sid in (1, 2, 3)}
            audios = {sid: os.path.join(paths["aud_dir"], f"scene_{sid:03d}.mp3")
                      for sid in (1, 2, 3)}
            with self.assertRaises(ValueError) as ctx_err:
                main.run_finalize_stage(ctx, fixture_scenes(), clips, audios)
            self.assertIn("scene 2", str(ctx_err.exception))
            self.assertFalse(os.path.isfile(paths["final"]))

    def test_missing_audio_fails_without_output(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            os.unlink(os.path.join(paths["aud_dir"], "scene_001.mp3"))
            ctx = make_ctx(tmp, paths)
            clips = {sid: os.path.join(paths["vid_dir"], f"scene_{sid:03d}.mp4")
                     for sid in (1, 2, 3)}
            audios = {sid: os.path.join(paths["aud_dir"], f"scene_{sid:03d}.mp3")
                      for sid in (1, 2, 3)}
            with self.assertRaises(ValueError) as ctx_err:
                main.run_finalize_stage(ctx, fixture_scenes(), clips, audios)
            self.assertIn("scene 1", str(ctx_err.exception))
            self.assertIn("narration", str(ctx_err.exception))
            self.assertFalse(os.path.isfile(paths["final"]))

    def test_valid_final_preserved_on_failure(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_project(tmp)
            ctx = make_ctx(tmp, paths)
            ctx["state"]["scene_ids"] = [1, 2, 3]
            clips = {sid: os.path.join(paths["vid_dir"], f"scene_{sid:03d}.mp4")
                     for sid in (1, 2, 3)}
            audios = {sid: os.path.join(paths["aud_dir"], f"scene_{sid:03d}.mp3")
                      for sid in (1, 2, 3)}
            main.run_finalize_stage(ctx, fixture_scenes(), clips, audios)
            good_bytes = Path(paths["final"]).read_bytes()
            self.assertTrue(st.valid_final_output(paths["final"]))
            # Break an input, rerun: must fail and leave final.mp4 untouched.
            with open(os.path.join(paths["vid_dir"], "scene_002.mp4"), "wb") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                main.run_finalize_stage(ctx, fixture_scenes(), clips, audios)
            self.assertEqual(Path(paths["final"]).read_bytes(), good_bytes)
            leftovers = []
            for directory in (paths["mux_dir"], tmp):
                leftovers += [f for f in os.listdir(directory) if ".tmp." in f]
            self.assertEqual(leftovers, [])


class FinalOutputValidationTest(unittest.TestCase):
    def test_video_only_is_not_valid_final(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            silent = os.path.join(tmp, "silent.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi",
                 "-i", "testsrc=duration=2:size=256x256:rate=10",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 "-f", "mp4", silent],
                check=True, capture_output=True)
            self.assertTrue(st.valid_video_file(silent))
            self.assertFalse(st.valid_final_output(silent))
            self.assertEqual(ff.probe_streams(silent), ["video"])

    def test_probe_streams_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.mp4")
            Path(bad).write_bytes(b"not a video file")
            self.assertEqual(ff.probe_streams(bad), [])
            self.assertFalse(st.valid_final_output(bad))
            self.assertFalse(st.valid_final_output(os.path.join(tmp, "missing.mp4")))


class ResumeFinalizeCommandTest(unittest.TestCase):
    """--resume --stage finalize through main.main: no regen, no network."""

    PROJECT = "finalize_cmd_tmp"

    def setUp(self):
        require_ffmpeg(self)
        base = os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT)
        shutil.rmtree(base, ignore_errors=True)
        self.addCleanup(shutil.rmtree, base, True)
        # Real project fixtures FIRST (before patches forbid storyboard writes).
        sb.save_storyboard(fixture_scenes(),
                           os.path.join(base, "storyboard.json"))
        for sub in ("videos", "audio", "muxed"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        for sid, seconds in zip(range(1, 4), DURATIONS):
            write_clip(os.path.join(base, "videos", f"scene_{sid:03d}.mp4"), seconds)
            write_audio(os.path.join(base, "audio", f"scene_{sid:03d}.mp3"), seconds)

        class NeverGenerate:
            def generate(self, request):
                raise AssertionError("must not regenerate previous stages")

        patches = [
            mock.patch.object(main, "create_provider",
                              side_effect=lambda kind, *_a, **_k: NeverGenerate()),
            mock.patch.object(main.ol, "check_reachable", return_value=None),
            mock.patch.object(ComfyClient, "check_reachable", return_value=None),
            mock.patch.object(main.ol, "generate_storyboard",
                              side_effect=AssertionError("no ollama")),
            mock.patch.object(main.sb, "save_storyboard",
                              side_effect=AssertionError("no save")),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_finalize_only(self):
        code = main.main(["--project", self.PROJECT, "--resume",
                          "--stage", "finalize"])
        self.assertEqual(code, 0)
        final = os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT, "final.mp4")
        self.assertTrue(st.valid_final_output(final))
        streams = ff.probe_streams(final)
        self.assertIn("video", streams)
        self.assertIn("audio", streams)
        final_dur = ff.probe_duration(final)
        self.assertAlmostEqual(final_dur, sum(DURATIONS), delta=1.0)
        self.assertEqual(audio_channels(final), 1)  # narration, not clip audio
        with open(os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT,
                               "state.json"), encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["stages"]["finalize"]["status"], "complete")
        # storyboard untouched (same scenes, same order).
        with open(os.path.join(AI_VIDEO_DIR, "projects", self.PROJECT,
                               "storyboard.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual([s["id"] for s in raw["scenes"]], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
