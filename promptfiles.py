"""Project-local sequential video prompt files.

Mode: projects/<project>/prompt/1.txt, 2.txt, ... -- each .txt file is the
COMPLETE prompt for exactly ONE text-to-video generation job
(1.txt -> scene_001.mp4, 2.txt -> scene_002.mp4, ...).

The file contents are passed to the video provider verbatim: never
summarized, condensed, planned, rewritten, or split (embedded "SCENE 1..N"
headings are model instructions, not pipeline scenes). No research, no
storyboard planner, no Qwen, no TTS input derivation.

Clip duration comes from the existing pipeline duration configuration
(config.yaml pipeline.duration); the prompt text is never parsed for
durations.

Version 2: an optional prompt/manifest.json ({"clips": {"1": {"duration":
10}, ...}}) overrides the per-clip duration. When absent, Version 1
behavior (uniform default duration) applies unchanged. Manifest clip ids
are filename stems ("1" <-> 1.txt); every discovered clip must have an
entry and vice versa (strict, no invented durations).
"""
from __future__ import annotations

import json
import math
import os
import re

PROMPT_EXTENSION = ".txt"
MANIFEST_FILENAME = "manifest.json"


def _natural_sort_key(path: str) -> list:
    """Split on digit runs so 2.txt sorts before 10.txt."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def _clip_number(name: str) -> int | None:
    """1-based clip number for valid prompt filenames, else None.

    Valid: positive-integer stem with a case-insensitive .txt extension
    (1.txt, 003.txt, 10.TXT). Anything else (abc.txt, scene1.txt,
    1_backup.txt, 0.txt) is not a numbered clip.
    """
    stem, ext = os.path.splitext(name)
    if ext.lower() != PROMPT_EXTENSION:
        return None
    if not re.fullmatch(r"[0-9]+", stem):
        return None
    number = int(stem)
    return number if number >= 1 else None


def discover_prompt_files(directory: str) -> tuple[list[tuple[int, str]], list[str]]:
    """Find numbered prompt files, natural-sorted by clip number.

    Returns (entries, ignored) where entries is [(clip_number, path)]
    sorted numerically and ignored lists non-clip filenames that were
    skipped (non-.txt, non-numeric stems, subdirectories are never
    considered). Gaps are allowed and reported as-is; nothing is invented.

    Raises FileNotFoundError when the directory is missing, ValueError
    when no valid prompt file exists or two files share one clip number.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Prompt directory not found: {directory}")
    entries: list[tuple[int, str]] = []
    ignored: list[str] = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        number = _clip_number(name)
        if number is None:
            if os.path.splitext(name)[1].lower() == PROMPT_EXTENSION:
                ignored.append(name)
            continue
        entries.append((number, path))
    if not entries:
        raise ValueError(f"No video prompt files found in {directory}. "
                         f"Add numbered prompts (1.txt, 2.txt, ...).")
    seen: dict[int, str] = {}
    for number, path in entries:
        if number in seen:
            raise ValueError(
                f"Duplicate prompt for clip {number}: "
                f"{os.path.basename(seen[number])} and "
                f"{os.path.basename(path)}.")
        seen[number] = path
    entries.sort(key=lambda item: item[0])
    ignored.sort(key=_natural_sort_key)
    return entries, ignored


def read_prompt_text(path: str) -> str:
    """Read one prompt file verbatim (UTF-8). Only file-level surrounding
    whitespace is stripped; internal content is preserved exactly.
    Raises ValueError naming the file when it is empty."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}. "
                         f"Each prompt file must contain one video prompt.")
    return text


def manifest_path(prompt_dir: str) -> str | None:
    """prompt/manifest.json when present, else None (Version 1 mode)."""
    candidate = os.path.join(prompt_dir, MANIFEST_FILENAME)
    return candidate if os.path.isfile(candidate) else None


def load_manifest(prompt_dir: str) -> dict[int, float] | None:
    """Read + strictly validate prompt/manifest.json.

    Returns {clip_number: duration} or None when no manifest exists
    (Version 1 compatibility: durations come from the pipeline default).

    Validation (ValueError, always naming the problem): malformed JSON,
    non-object root, missing/non-object "clips", non-numeric clip keys,
    non-positive/non-finite/non-numeric durations (bools rejected).
    Backend-specific ranges are NOT checked here; the video provider
    enforces its own constraints at generation time.
    """
    path = manifest_path(prompt_dir)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Manifest is invalid JSON ({path}): {exc}")
    if not isinstance(raw, dict) or not isinstance(raw.get("clips"), dict):
        raise ValueError(
            f"Manifest must contain a clips object ({path}); expected "
            f'{{"clips": {{"1": {{"duration": 10}}, ...}}}}.')
    durations: dict[int, float] = {}
    for key, entry in raw["clips"].items():
        if not isinstance(key, str) or not re.fullmatch(r"[0-9]+", key):
            raise ValueError(
                f"Manifest has non-numeric clip id {key!r} ({path}); "
                f"clip ids must match prompt filename stems (\"1\", \"2\", ...).")
        number = int(key)
        if number < 1:
            raise ValueError(
                f"Manifest has invalid clip id {key!r} ({path}); "
                f"clip numbers start at 1.")
        if number in durations:
            raise ValueError(
                f"Manifest has a duplicate entry for clip {number} ({path}).")
        duration = entry.get("duration") if isinstance(entry, dict) else None
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) \
                or not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Manifest has an invalid duration for clip {number} "
                f"({path}); duration must be a positive number of seconds.")
        durations[number] = duration
    return durations


def load_prompt_scenes(prompt_dir: str, default_duration: float) -> list[dict]:
    """Build the minimal scene representation from prompt files.

    One file = one scene: {"id", "duration", "image_prompt", "video_prompt",
    "narration", "prompt_file"}. Without a manifest, every scene uses
    default_duration (Version 1). With prompt/manifest.json, each scene
    takes clips["<id>"].duration; the manifest must cover exactly the
    discovered clips (missing entries and extra entries both fail loudly
    before any generation). Narration is always "" (video prompts are never
    narration; the audio stage behaves exactly as it would for narration
    without text under existing validation rules).
    """
    entries, _ignored = discover_prompt_files(prompt_dir)
    manifest = load_manifest(prompt_dir)
    if manifest is not None:
        found = {number for number, _ in entries}
        missing = sorted(found - set(manifest))
        if missing:
            raise ValueError(
                f"Manifest is missing duration for "
                f"{', '.join(f'clip {n}' for n in missing)} "
                f"({os.path.join(prompt_dir, MANIFEST_FILENAME)}).")
        extra = sorted(set(manifest) - found)
        if extra:
            raise ValueError(
                f"Manifest has entries for missing prompt files: "
                f"{', '.join(f'clip {n}' for n in extra)} "
                f"({os.path.join(prompt_dir, MANIFEST_FILENAME)}).")
    scenes = []
    for number, path in entries:
        scenes.append({"id": number,
                       "duration": manifest[number] if manifest is not None
                       else default_duration,
                       "image_prompt": "",
                       "video_prompt": read_prompt_text(path),
                       "narration": "",
                       "prompt_file": os.path.basename(path)})
    return scenes
