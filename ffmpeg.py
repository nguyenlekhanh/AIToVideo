"""FFmpeg helpers: mux each scene (video + narration), then concat scenes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile


def check_ffmpeg(executable: str = "ffmpeg") -> str:
    path = shutil.which(executable) or (executable if os.path.isfile(executable) else None)
    if path is None:
        raise RuntimeError(
            f"ffmpeg executable not found: {executable!r}. Install ffmpeg and add it to PATH.")
    return path


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed: {' '.join(cmd)}\n{exc.stderr[-3000:]}") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg executable not found: {cmd[0]}") from exc


def probe_duration(path: str, ffprobe_exe: str = "ffprobe") -> float:
    exe = shutil.which(ffprobe_exe) or ffprobe_exe
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            check=True, capture_output=True, text=True).stdout
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def count_frames(path: str, ffprobe_exe: str = "ffprobe") -> int:
    """Number of frames in the first video stream, or 0 when unreadable."""
    exe = shutil.which(ffprobe_exe) or ffprobe_exe
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
             path],
            check=True, capture_output=True, text=True).stdout
        return max(0, int(out.strip().split()[0]))
    except Exception:
        return 0


def extract_last_frame(video_path: str, out_path: str,
                       executable: str = "ffmpeg") -> str:
    """Extract the ACTUAL final frame of a clip (deterministic PNG).

    Counts frames with ffprobe, then selects frame N-1 exactly. Falls back
    to `-sseof` only when the count is unavailable. Used for chained
    generation where clip N+1 must start from clip N's true ending.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    total = count_frames(video_path)
    if total > 0:
        run([executable, "-y", "-i", video_path, "-vf",
             f"select='eq(n\\,{total - 1})'", "-vframes", "1", out_path])
    else:
        run([executable, "-y", "-sseof", "-0.1", "-i", video_path,
             "-vframes", "1", out_path])
    return out_path


def probe_streams(path: str, ffprobe_exe: str = "ffprobe") -> list[str]:
    """Codec types present in the file (e.g. ["video", "audio"]).

    Returns [] when the file is unreadable. Used to verify a muxed output
    actually contains both a video and an audio stream.
    """
    exe = shutil.which(ffprobe_exe) or ffprobe_exe
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "json", path],
            check=True, capture_output=True, text=True).stdout
        return [str(s.get("codec_type", ""))
                for s in json.loads(out).get("streams", [])]
    except Exception:
        return []


def mux_scene(video_path: str, audio_path: str, out_path: str,
              executable: str = "ffmpeg", crf: int = 19, preset: str = "veryfast",
              copy_video: bool = False) -> str:
    """Combine one silent clip with its narration.

    The clip is looped with a FINITE loop count derived from probed
    durations (never `-stream_loop -1`: with some audio files `-shortest`
    never fires on an infinite input and the mux runs away). `-shortest`
    then trims the small overshoot.

    With copy_video=True the video stream is stream-copied (`-c:v copy`)
    instead of re-encoded (faster; requires compatible input codec).
    """
    import math
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    video_dur = probe_duration(video_path)
    audio_dur = probe_duration(audio_path)
    loops = 0
    if video_dur > 0 and audio_dur > video_dur:
        loops = math.ceil(audio_dur / video_dur) - 1
    cmd = [
        executable, "-y",
        "-stream_loop", str(loops), "-i", video_path,
        "-i", audio_path,
        # Explicit mapping: without -map, ffmpeg's automatic stream
        # selection prefers the clip's embedded stereo track over the mono
        # narration (most channels wins) and silently drops the narration.
        "-map", "0:v:0", "-map", "1:a:0",
    ]
    if audio_dur > 0:
        cmd += ["-t", f"{audio_dur:.3f}"]
    if copy_video:
        cmd += ["-shortest", "-max_interleave_delta", "100M",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", out_path]
    else:
        cmd += ["-shortest", "-max_interleave_delta", "100M",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", str(crf), "-preset", preset,
                "-c:a", "aac", "-b:a", "128k", out_path]
    run(cmd)
    return out_path


def concat_scenes(scene_paths: list[str], out_path: str,
                  executable: str = "ffmpeg", copy: bool = False) -> str:
    """Concatenate per-scene MP4s in order.

    Default re-encodes for robustness across differing inputs. With
    copy=True the streams are concatenated without re-encoding (faster;
    requires identical codecs/parameters in all inputs).
    """
    if not scene_paths:
        raise ValueError("No scenes to concatenate.")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        for p in scene_paths:
            # concat demuxer: escape single quotes
            f.write(f"file '{os.path.abspath(p).replace(chr(39), chr(39)+chr(39))}'\n")
        list_file = f.name
    try:
        if copy:
            run([executable, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c", "copy", out_path])
        else:
            run([executable, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "veryfast",
                 "-c:a", "aac", "-b:a", "128k", out_path])
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass
    return out_path
