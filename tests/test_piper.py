"""Unit tests for the Piper local TTS provider.

No network: Piper itself is faked via monkeypatched subprocess (a fake
binary writes a real WAV, converted by the real local ffmpeg). The real
Piper binary/models are never required. Follows tests/test_resume.py
conventions (require_ffmpeg guard, tmp dirs).
"""
from __future__ import annotations

import builtins
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import main
import state as st
from providers.audio.base import AudioRequest
from providers.audio.edge_tts import EdgeTtsProvider
from providers.audio.piper_tts import PiperAudioProvider
from providers.errors import ProviderError
from providers.registry import create_provider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(AI_VIDEO_DIR, "voices", "piper", "vn")
VI_SENTENCE = "Vì sao Việt Nam lại phát triển nhanh như vậy?"
FAKE_EXE = os.path.join("C:", os.sep, "fake", "piper.exe")


def require_ffmpeg(testcase):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        testcase.skipTest("ffmpeg/ffprobe not available")


def write_wav(path, seconds=0.3, framerate=22050):
    n = int(seconds * framerate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def make_provider(**overrides):
    settings = {"voice_dir_path": VOICES_DIR, "voice_dir": "voices/piper/vn",
                "executable": FAKE_EXE}
    settings.update(overrides)
    return PiperAudioProvider(name="piper", settings=settings)


def make_provider_with_binary(tmpdir):
    """Provider whose configured executable exists (never executed: mocked)."""
    exe = os.path.join(tmpdir, "piper.exe")
    Path(exe).write_bytes(b"fake-binary")
    return make_provider(executable=exe)


def require_ffmpeg(testcase):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        testcase.skipTest("ffmpeg/ffprobe not available")


class FakePiper:
    """Simulates the piper binary; records exact stdin bytes and argv."""

    def __init__(self, returncode=0, stderr=b"", wav_seconds=0.3):
        self.returncode = returncode
        self.stderr = stderr
        self.wav_seconds = wav_seconds
        self.inputs = []
        self.commands = []
        # Captured at construction (before mock.patch is applied).
        self._real_run = subprocess.run
        self._real_run = subprocess.run

    def __call__(self, cmd, **kwargs):
        if "--model" in cmd and "--output_file" in cmd:
            self.commands.append(cmd)
            self.inputs.append(kwargs.get("input", b""))
            out = cmd[cmd.index("--output_file") + 1]
            write_wav(out, seconds=self.wav_seconds)
            return subprocess.CompletedProcess(cmd, self.returncode,
                                               stdout=b"", stderr=self.stderr)
        return self._real_run(cmd, **kwargs)


def patched_subprocess(fake):
    return mock.patch("subprocess.run", side_effect=fake)


class RegistryTest(unittest.TestCase):
    def test_piper_selected(self):
        provider = create_provider("audio", AI_VIDEO_DIR, "piper")
        self.assertIsInstance(provider, PiperAudioProvider)

    def test_relative_executable_resolves_against_app_root(self):
        from providers.registry import lookup
        entry = lookup(AI_VIDEO_DIR, "audio", "piper")
        expected = os.path.join(AI_VIDEO_DIR, "tools", "piper", "piper.exe")
        self.assertEqual(entry["executable_path"], expected)
        provider = create_provider("audio", AI_VIDEO_DIR, "piper")
        self.assertEqual(provider.resolve_executable(), expected)
        self.assertTrue(os.path.isfile(expected))

    def test_edge_alias_selected(self):
        provider = create_provider("audio", AI_VIDEO_DIR, "edge")
        self.assertIsInstance(provider, EdgeTtsProvider)

    def test_edge_tts_key_unchanged(self):
        provider = create_provider("audio", AI_VIDEO_DIR, "edge_tts")
        self.assertIsInstance(provider, EdgeTtsProvider)


class VoiceResolutionTest(unittest.TestCase):
    def test_exact_filename_stem(self):
        provider = make_provider()
        model, config = provider.resolve_voice("ngochuyen")
        self.assertEqual(model, os.path.join(VOICES_DIR, "ngochuyen.onnx"))
        self.assertEqual(config, os.path.join(VOICES_DIR, "ngochuyen.onnx.json"))

    def test_missing_json_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(os.path.join(tmp, "x.onnx")).write_bytes(b"fake")
            provider = make_provider(voice_dir_path=tmp, voice_dir=tmp)
            with self.assertRaises(ProviderError) as ctx:
                provider.resolve_voice("x")
            self.assertIn("Piper voice config not found", str(ctx.exception))
            self.assertIn("x.onnx.json", str(ctx.exception))

    def test_no_alias_lookup(self):
        provider = make_provider()
        for name in ("Ngoc Huyen", "NGOCHUYEN", "Ngọc Huyền"):
            with self.subTest(name=name):
                with self.assertRaises(ProviderError) as ctx:
                    provider.resolve_voice(name)
                self.assertIn("Piper voice not found", str(ctx.exception))

    def test_no_fuzzy_matching(self):
        provider = make_provider()
        for name in ("ngochuye", "ngochuyenn", "gnochuyen", "banma"):
            with self.subTest(name=name):
                with self.assertRaises(ProviderError) as ctx:
                    provider.resolve_voice(name)
                self.assertIn("Piper voice not found", str(ctx.exception))

    def test_missing_voice_message_shows_configured_dir(self):
        provider = make_provider()
        with self.assertRaises(ProviderError) as ctx:
            provider.resolve_voice("nosuchvoice")
        message = str(ctx.exception)
        self.assertIn("Piper voice not found", message)
        self.assertIn(os.path.join("voices", "piper", "vn",
                                   "nosuchvoice.onnx"), message)


class GenerationTest(unittest.TestCase):
    def test_utf8_narration_reaches_piper_unchanged(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            fake = FakePiper()
            out = os.path.join(tmp, "scene_001.mp3")
            with patched_subprocess(fake):
                result = provider.generate(AudioRequest(
                    text=VI_SENTENCE, output_path=Path(out), voice="ngochuyen"))
            self.assertEqual(fake.inputs, [VI_SENTENCE.encode("utf-8")])
            self.assertEqual(str(result.path), out)
            self.assertTrue(st.valid_audio_file(out))
            import ffmpeg as ff
            self.assertGreater(ff.probe_duration(out), 0)
            leftovers = [f for f in os.listdir(tmp)
                         if f.endswith((".wav", ".conv.mp3"))]
            self.assertEqual(leftovers, [])

    def test_output_at_expected_scene_path(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            fake = FakePiper()
            out = os.path.join(tmp, "audio", "scene_002.mp3")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with patched_subprocess(fake):
                provider.generate(AudioRequest(
                    text="Xin chào.", output_path=Path(out), voice="banmai"))
            self.assertTrue(os.path.isfile(out))

    def test_empty_text_rejected(self):
        provider = make_provider()
        with self.assertRaises(ProviderError):
            provider.generate(AudioRequest(text="  ", output_path=Path("x.mp3"),
                                           voice="ngochuyen"))

    def test_missing_executable_fails_clearly(self):
        provider = make_provider(executable=os.path.join(
            "C:", os.sep, "definitely", "not", "here", "piper.exe"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(AudioRequest(
                    text="Xin chào.", output_path=Path(tmp) / "x.mp3",
                    voice="ngochuyen"))
            self.assertIn("Piper executable not found", str(ctx.exception))
            self.assertIn("Configure the Piper executable path", str(ctx.exception))

    def test_missing_model_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(AudioRequest(
                    text="Xin chào.", output_path=Path(tmp) / "x.mp3",
                    voice="nosuchvoice"))
            self.assertIn("Piper voice not found", str(ctx.exception))

    def test_piper_nonzero_exit_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            fake = FakePiper(returncode=1, stderr=b"piper: boom")
            with patched_subprocess(fake):
                with self.assertRaises(ProviderError) as ctx:
                    provider.generate(AudioRequest(
                        text="Xin chào.", output_path=Path(tmp) / "x.mp3",
                        voice="ngochuyen"))
            message = str(ctx.exception)
            self.assertIn("exit 1", message)
            self.assertIn("boom", message)
            self.assertFalse(os.path.isfile(os.path.join(tmp, "x.mp3")))
            leftovers = [f for f in os.listdir(tmp)
                         if f.endswith((".wav", ".conv.mp3"))]
            self.assertEqual(leftovers, [])


class IsolationTest(unittest.TestCase):
    def test_no_network_or_edge_usage_in_source(self):
        with open(os.path.join(AI_VIDEO_DIR, "providers", "audio",
                               "piper_tts.py"), encoding="utf-8") as f:
            source = f.read()
        for banned in ("urllib", "socket", "requests", "http.client", "httpx",
                       "aiohttp", "urlopen", "edge_tts", "Communicate"):
            self.assertNotIn(banned, source)

    def test_does_not_import_edge_tts(self):
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "edge_tts" or name.startswith("edge_tts."):
                raise AssertionError("Piper must not import edge_tts")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            fake = FakePiper()
            with patched_subprocess(fake), \
                 mock.patch("builtins.__import__", side_effect=guard):
                provider.generate(AudioRequest(
                    text="Xin chào.", output_path=Path(tmp) / "x.mp3",
                    voice="ngochuyen"))


class AudioStageIntegrationTest(unittest.TestCase):
    def test_run_audio_stage_with_piper(self):
        require_ffmpeg(self)
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider_with_binary(tmp)
            fake = FakePiper()
            aud_dir = os.path.join(tmp, "audio")
            os.makedirs(aud_dir, exist_ok=True)
            state_path = os.path.join(tmp, "state.json")
            ctx = {"audio_provider": provider, "state": st.new_state("test"),
                   "state_path": state_path,
                   "aud_dir": aud_dir, "voice": "ngochuyen",
                   "rate": "+0%", "pitch": "+0Hz", "project": "test"}
            scenes = [{"id": 1, "duration": 5, "image_prompt": "p",
                       "video_prompt": "v", "narration": VI_SENTENCE}]
            with patched_subprocess(fake):
                paths = main.run_audio_stage(ctx, scenes)
            dest = os.path.join(aud_dir, "scene_001.mp3")
            self.assertEqual(paths, {1: dest})
            self.assertTrue(st.valid_audio_file(dest))
            self.assertEqual(fake.inputs, [VI_SENTENCE.encode("utf-8")])


class CliTest(unittest.TestCase):
    def test_audio_model_piper_parses(self):
        args = main.parse_args(["hello", "--audio-model", "piper",
                                "--voice", "ngochuyen"])
        self.assertEqual(args.audio_model, "piper")
        self.assertEqual(args.voice, "ngochuyen")

    def test_audio_model_edge_parses(self):
        args = main.parse_args(["hello", "--audio-model", "edge"])
        self.assertEqual(args.audio_model, "edge")


if __name__ == "__main__":
    unittest.main()
