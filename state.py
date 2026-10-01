"""Lightweight stage progress persistence + output validation.

state.json records per-scene completion; it never replaces filesystem
truth: on resume every claimed output is re-validated (exists, readable,
non-zero size, valid format, plausible duration). Missing/corrupt files
are regenerated. Missing/corrupt state.json starts fresh, never crashes.

Schema:
{
  "version": 1,
  "project": "...",
  "base_seed": 123 | null,
  "scene_ids": [1, 2, 3],
  "stages": {
    "research": {"status": "complete"|"pending", "attempts": 0},
    "storyboard": {...},
    "image": {"status": ..., "completed_scenes": [...], "attempts": {"1": 2}},
    "video": {...},
    "audio": {...},
    "finalize": {"status": ..., "attempts": 0}
  },
  "updated_at": "iso-8601"
}
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import ffmpeg as ff

STATE_VERSION = 1
SCENE_STAGES = ("image", "video", "audio")
SINGLE_STAGES = ("research", "storyboard", "finalize")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def new_state(project: str) -> dict:
    state = {
        "version": STATE_VERSION,
        "project": project,
        "base_seed": None,
        "scene_ids": [],
        "stages": {},
        "updated_at": utcnow_iso(),
    }
    for stage in SINGLE_STAGES:
        state["stages"][stage] = {"status": "pending", "attempts": 0}
    for stage in SCENE_STAGES:
        state["stages"][stage] = {"status": "pending", "completed_scenes": [],
                                  "attempts": {}}
    return state


def load_state(path: str, project: str = "") -> dict:
    """Load state; missing/corrupt/wrong-shape files yield a fresh skeleton."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
            raise ValueError("bad shape")
        state = new_state(project or data.get("project", ""))
        # Merge defensively: keep known structure, adopt stored values.
        for stage, entry in data["stages"].items():
            if stage in state["stages"] and isinstance(entry, dict):
                state["stages"][stage].update(entry)
        for key in ("base_seed", "scene_ids", "updated_at"):
            if key in data:
                state[key] = data[key]
        if not isinstance(state.get("scene_ids"), list):
            state["scene_ids"] = []
        return state
    except (OSError, ValueError):
        return new_state(project)
    except Exception:
        return new_state(project)


def save_state(path: str, state: dict) -> None:
    """Atomic write (tmp + rename) so Ctrl+C can never corrupt progress."""
    state["updated_at"] = utcnow_iso()
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def _refresh_status(state: dict, stage: str) -> None:
    if stage in SCENE_STAGES:
        total = set(state.get("scene_ids", []))
        done = set(state["stages"][stage].get("completed_scenes", []))
        if total and done >= total:
            status = "complete"
        elif done:
            status = "partial"
        else:
            status = "pending"
        state["stages"][stage]["status"] = status


def mark_scene_complete(state: dict, stage: str, scene_id: int,
                        extra: dict | None = None) -> None:
    """Record a validated scene output. Never call before validating output."""
    entry = state["stages"][stage]
    completed = entry.setdefault("completed_scenes", [])
    if scene_id not in completed:
        completed.append(scene_id)
        completed.sort()
    attempts = entry.setdefault("attempts", {})
    attempts[str(scene_id)] = int(attempts.get(str(scene_id), 0)) + 1
    if extra:
        entry.setdefault("meta", {})[str(scene_id)] = extra
    _refresh_status(state, stage)


def mark_stage_complete(state: dict, stage: str) -> None:
    entry = state["stages"][stage]
    entry["status"] = "complete"
    entry["attempts"] = int(entry.get("attempts", 0)) + 1


def completed_scenes(state: dict, stage: str) -> list[int]:
    return list(state.get("stages", {}).get(stage, {}).get("completed_scenes", []))


def scene_attempts(state: dict, stage: str, scene_id: int) -> int:
    return int(state.get("stages", {}).get(stage, {}).get("attempts", {}).get(
        str(scene_id), 0))


def stage_status(state: dict, stage: str) -> str:
    _refresh_status(state, stage) if stage in SCENE_STAGES else None
    return str(state.get("stages", {}).get(stage, {}).get("status", "pending"))


# -- output validation (filesystem truth, not state truth) --

def _readable_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def valid_image_file(path: str) -> bool:
    """PNG/JPEG magic + non-zero size."""
    if not _readable_nonempty(path):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head[:8] == PNG_MAGIC or head[:2] == JPEG_MAGIC
    except OSError:
        return False


def valid_audio_file(path: str) -> bool:
    """Exists, readable, non-zero size."""
    return _readable_nonempty(path)


def probe_video_duration(path: str) -> float | None:
    """Probed media duration in seconds, or None when unreadable."""
    try:
        duration = ff.probe_duration(path)
    except Exception:
        return None
    return duration if duration > 0 else None


def valid_final_output(path: str) -> bool:
    """Final-output gate: a probed video file containing BOTH a video
    stream and an audio stream. A video-only file is not a valid final."""
    if not valid_video_file(path):
        return False
    streams = ff.probe_streams(path)
    return "video" in streams and "audio" in streams


def valid_video_file(path: str, expected_duration: float | None = None) -> bool:
    """Exists, non-zero, ffprobe-readable; optionally duration-plausible.

    Duration tolerance is max(1.0s, 15%): mismatches mark the file
    incomplete (regenerate) rather than failing loudly.
    """
    if not _readable_nonempty(path):
        return False
    actual = probe_video_duration(path)
    if actual is None:
        return False
    if expected_duration is not None:
        try:
            expected = float(expected_duration)
        except (TypeError, ValueError):
            return False
        if expected <= 0:
            return False
        if abs(actual - expected) > max(1.0, 0.15 * expected):
            return False
    return True
