"""Piper local TTS narration provider (one file per scene).

Runs the Piper executable locally (no network, no fallback to Edge TTS).
The voice name maps DIRECTLY to a model filename stem: ``--voice ngochuyen``
resolves ``<voice_dir>/ngochuyen.onnx`` plus its ``.onnx.json`` sidecar.
No aliases, no normalization, no fuzzy matching.

Piper writes WAV; the file is converted to MP3 with the existing ffmpeg
utilities so output stays compatible with the finalize stage.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import ffmpeg as ff

from ..errors import ProviderError
from .base import AudioProvider, AudioRequest, AudioResult


class PiperAudioProvider(AudioProvider):
    provider_id = "piper"

    def __init__(self, name: str = "piper", settings: dict | None = None,
                 **kwargs):
        self.name = name
        self.settings = settings or {}

    def resolve_executable(self) -> str:
        """Absolute path to the piper binary (configured path or PATH)."""
        configured = (self.settings.get("executable_path")
                      or self.settings.get("executable") or "").strip()
        if configured:
            if os.path.isabs(configured):
                if os.path.isfile(configured):
                    return configured
            else:
                found = shutil.which(configured)
                if found:
                    return found
            raise ProviderError(
                self.provider_id,
                "Piper executable not found. Configure the Piper executable path.")
        found = shutil.which("piper")
        if found:
            return found
        raise ProviderError(
            self.provider_id,
            "Piper executable not found. Configure the Piper executable path.")

    def _voice_dir(self) -> tuple[str, str]:
        """Return (absolute_dir, display_dir) for voice model lookup."""
        absolute = (self.settings.get("voice_dir_path") or "").strip()
        if absolute:
            return absolute, absolute
        configured = (self.settings.get("voice_dir") or "").strip()
        if configured and os.path.isabs(configured):
            return configured, configured
        if configured:
            return "", configured
        return "", "voices/piper/vn"

    def resolve_voice(self, voice: str) -> tuple[str, str]:
        """Map a voice name DIRECTLY to its model + config files.

        No case folding, no accent stripping, no aliases, no fuzzy matching:
        ``ngochuyen`` means exactly ``<voice_dir>/ngochuyen.onnx``.
        The match is checked against the actual directory listing so the
        behavior is identical on case-insensitive and case-sensitive
        filesystems.
        """
        voice = (voice or "").strip()
        voice_dir, display_dir = self._voice_dir()
        if not voice_dir:
            raise ProviderError(
                self.provider_id,
                "Piper voice directory is not configured.")
        wanted_model = voice + ".onnx"
        try:
            present = set(os.listdir(voice_dir))
        except OSError as exc:
            raise ProviderError(
                self.provider_id,
                f"Piper voice directory unreadable: {voice_dir} ({exc})") from exc
        display_model = os.path.join(display_dir, voice + ".onnx")
        if wanted_model not in present:
            raise ProviderError(
                self.provider_id,
                f"Piper voice not found: {display_model}")
        model = os.path.join(voice_dir, wanted_model)
        config = model + ".json"
        if (wanted_model + ".json") not in present:
            raise ProviderError(
                self.provider_id,
                f"Piper voice config not found: {display_model}.json")
        return model, config

    def generate(self, request: AudioRequest) -> AudioResult:
        text = (request.text or "").strip()
        if not text:
            raise ProviderError(self.provider_id, "Cannot synthesize empty narration.")
        executable = self.resolve_executable()
        model, _config = self.resolve_voice(request.voice)
        os.makedirs(os.path.dirname(os.path.abspath(request.output_path)), exist_ok=True)
        # The stage passes a ".tmp" path (no usable extension); ffmpeg needs
        # a real container extension, so convert to a sibling .mp3 temp file
        # first and rename into place only on success.
        wav_tmp = str(request.output_path) + ".piper.wav"
        mp3_tmp = str(request.output_path) + ".conv.mp3"
        try:
            proc = subprocess.run(
                [executable, "--model", model, "--output_file", wav_tmp],
                input=text.encode("utf-8"),
                capture_output=True)
            if proc.returncode != 0:
                raise ProviderError(
                    self.provider_id,
                    f"Piper synthesis failed (exit {proc.returncode}): "
                    f"{proc.stderr.decode('utf-8', 'replace')[:1000]}")
            if not os.path.isfile(wav_tmp) or os.path.getsize(wav_tmp) == 0:
                raise ProviderError(
                    self.provider_id,
                    "Piper synthesis failed: no WAV output produced.")
            try:
                ffmpeg_exe = ff.check_ffmpeg()
            except RuntimeError as exc:
                raise ProviderError(self.provider_id, str(exc)) from exc
            try:
                ff.run([ffmpeg_exe, "-y", "-i", wav_tmp,
                        "-c:a", "libmp3lame", "-b:a", "128k", mp3_tmp])
            except RuntimeError as exc:
                raise ProviderError(
                    self.provider_id,
                    f"Piper MP3 conversion failed: {exc}") from exc
            if not os.path.isfile(mp3_tmp) or os.path.getsize(mp3_tmp) == 0:
                raise ProviderError(
                    self.provider_id,
                    "Piper MP3 conversion failed: no MP3 output produced.")
            os.replace(mp3_tmp, str(request.output_path))
        finally:
            for temp in (wav_tmp, mp3_tmp):
                try:
                    if os.path.isfile(temp):
                        os.unlink(temp)
                except OSError:
                    pass
        return AudioResult(path=Path(request.output_path))
