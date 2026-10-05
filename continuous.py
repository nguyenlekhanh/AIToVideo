"""Continuous Video Chain: one image + one prompt -> chained clips.

Orchestration layer only (no backend logic here). Qwen (text-only, never
sent images) plans a continuous action storyboard; the engine then executes
clips sequentially through the EXISTING VideoProvider abstraction
(LTX / Wan / Wan Animate2 / future backends):

    source image + clip 1 prompt -> clip_001.mp4 -> lastframe_001.png
    lastframe_001.png + clip 2 prompt -> clip_002.mp4 -> lastframe_002.png
    ...

Clip N+1 ALWAYS starts from clip N's true last frame, never from the
original image (except clip 1). Resume is filesystem-truth with a
per-clip manifest recording the start-frame hash, so a regenerated clip
automatically cascades to stale downstream clips.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path as _Path

import ffmpeg as ff
import ollama as ol
import state as st
from providers.video.base import VideoRequest

MIN_CLIP_DURATION = 3
MAX_CLIP_DURATION = 15
# Planner freedom when the user gives neither --clips nor --duration.
MIN_AUTO_CLIPS = 2
MAX_AUTO_CLIPS = 8
# Sensible default clip length used to derive counts from --duration.
NOMINAL_CLIP_DURATION = 7
MAX_ATTEMPTS = 3

BOARD_FILENAME = "continuous_storyboard.json"
CLIPS_DIRNAME = "continuous"
MANIFEST_FILENAME = "manifest.json"
FINAL_FILENAME = "final.mp4"
IDENTITY_FILENAME = "identity.json"
MEMORY_FILENAME = "memory.json"
IDENTITY_VERSION = 2
MEMORY_VERSION = 1
# Project-local character references: <project_dir>/ImageCharacterReference/.
# Never a global/shared directory; each project owns its references.
CHARACTER_REF_DIRNAME = "ImageCharacterReference"
CHARACTER_ID_RE = re.compile(r"^character_(\d+)$", re.IGNORECASE)
CHARACTER_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

CONTINUITY_BLOCK = """Continue directly from the provided starting frame.
The starting frame is the exact final frame of the previous video clip.
Preserve: character identity, face, hair, clothing, body proportions,
environment, lighting, time of day, visual style, spatial continuity.
Do not restart the scene. Continue the existing motion naturally.

SCENE ACTION:
"""

CHAIN_SYSTEM_PROMPT = """You are an action director planning ONE continuous video from a single user prompt and a single starting image.
Divide the described action into a sequence of short clips that form one unbroken timeline. Cut ONLY at meaningful motion boundaries (an action starts / a movement completes / direction changes / a new beat begins). Never repeat the same generic action in every clip; each clip must advance the timeline.

Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"clips": [{"id": 1, "duration": 7, "start_state": "...", "action": "...", "end_state": "...", "video_prompt": "..."}]}

Rules:
- ids are 1..N in order, one entry per clip, exactly the requested count.
- duration is an integer 3..15 seconds, chosen from action complexity (vary it; a quick beat is short, a complex move is long).
- start_state: how the clip begins. Clip 1 starts from the given image ("as shown in the starting image, ..."). Every later clip's start_state MUST continue the previous clip's end_state (same pose-in-progress, same positions, same camera) - never restart the action.
- action: the single concrete beat of this clip (who moves where, how, and what changes). Track every important character separately (position, clothing, weapon/object ownership, injuries); never swap identities between clips.
- end_state: the exact frozen moment this clip ends on (pose, positions, camera) - this is what the next clip continues from.
- video_prompt: the image-to-video prompt for this clip (camera move + subject motion + environment motion), consistent with start_state/action/end_state. Pure visual prose.
- Preserve character identity (face, hair, clothing, proportions, colors), environment (location, time of day, weather, lighting, objects, style) and camera language across clips unless the story explicitly changes them. State changes (a fall, an injury, a costume change, day turning to night) happen ONCE at the right beat and persist afterwards.
- ANTI-RESET (mandatory): clip N begins from the exact final frame of clip N-1. Do not restart the action. Do not reset any pose. Do not teleport characters. Do not change clothing, environment, object ownership, or character roles unless the story explicitly requires it at this beat. Continue the physical action naturally.
- You may also include an "initial_state" object (global/characters/objects/camera describing the opening frame) and a per-clip "state" object (the same keys, describing the world at the END of that clip). Keep them short, concrete, and consistent with the clip texts; omit anything uncertain. Example: {"initial_state": {"global": {"location": "...", "time": "...", "weather": "...", "lighting": "..."}, "characters": [{"id": "character_a", "position": "...", "facing": "...", "clothing_state": "unchanged", "props": [], "condition": "uninjured"}], "objects": [], "camera": {"shot": "...", "movement": "...", "direction": "..."}}}.
- Ground everything in the user prompt. Do not invent unrelated characters, locations, or events."""


# -- paths ---------------------------------------------------------------

def chain_paths(base_dir: str) -> dict:
    """All continuous-chain artifact locations under a project directory."""
    out_dir = os.path.join(base_dir, CLIPS_DIRNAME)
    return {
        "board": os.path.join(base_dir, BOARD_FILENAME),
        "out_dir": out_dir,
        "manifest": os.path.join(out_dir, MANIFEST_FILENAME),
        "final": os.path.join(out_dir, FINAL_FILENAME),
    }


def clip_dir(out_dir: str, clip_id: int) -> str:
    return os.path.join(out_dir, f"{clip_id:03d}")


def start_path(out_dir: str, clip_id: int) -> str:
    return os.path.join(clip_dir(out_dir, clip_id), "start.png")


def video_path(out_dir: str, clip_id: int) -> str:
    return os.path.join(clip_dir(out_dir, clip_id), "video.mp4")


def lastframe_path(out_dir: str, clip_id: int) -> str:
    return os.path.join(clip_dir(out_dir, clip_id), "lastframe.png")


def prompt_path(out_dir: str, clip_id: int) -> str:
    return os.path.join(clip_dir(out_dir, clip_id), "prompt.txt")


def file_hash(path: str) -> str | None:
    """sha256 of file bytes; None when unreadable. Tracks start-frame deps."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# -- planner ---------------------------------------------------------------

def _validate_chain(parsed: dict, num_clips: int | None,
                    total_duration: float | None) -> dict:
    """Structural gate for the planner output. Never fabricates."""
    if not isinstance(parsed, dict):
        raise ValueError("Planner did not return a JSON object.")
    clips_raw = parsed.get("clips")
    if not isinstance(clips_raw, list) or not clips_raw:
        raise ValueError("Planner returned no clips.")
    if num_clips is not None and len(clips_raw) != num_clips:
        raise ValueError(
            f"Planner returned {len(clips_raw)} clips, "
            f"expected exactly {num_clips}.")
    if num_clips is None and not MIN_AUTO_CLIPS <= len(clips_raw) <= 12:
        raise ValueError(
            f"Planner returned {len(clips_raw)} clips "
            f"(allowed {MIN_AUTO_CLIPS}..12).")
    clips = []
    for i, raw in enumerate(clips_raw, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Clip #{i} is not an object.")
        for field in ("start_state", "action", "end_state", "video_prompt"):
            value = raw.get(field, "")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Clip #{i} has empty {field!r}.")
        char_ids = raw.get("characters", [])
        if char_ids is None:
            char_ids = []
        if not isinstance(char_ids, list) or any(
                not isinstance(c, str) or not c.strip() for c in char_ids):
            raise ValueError(
                f"Clip #{i} 'characters' must be a list of non-empty "
                f"strings.")
        try:
            duration = int(raw.get("duration", 0))
        except (TypeError, ValueError):
            raise ValueError(f"Clip #{i} has non-integer duration.")
        if not MIN_CLIP_DURATION <= duration <= MAX_CLIP_DURATION:
            raise ValueError(
                f"Clip #{i} duration {duration} outside "
                f"{MIN_CLIP_DURATION}..{MAX_CLIP_DURATION}.")
        clips.append({
            "id": i,
            "duration": duration,
            "characters": [str(c).strip() for c in char_ids],
            "start_state": raw["start_state"].strip(),
            "action": raw["action"].strip(),
            "end_state": raw["end_state"].strip(),
            "video_prompt": raw["video_prompt"].strip(),
            "state": _normalize_state_block(raw.get("state")),
        })
    if total_duration is not None:
        planned = sum(c["duration"] for c in clips)
        tolerance = max(5.0, 0.25 * total_duration)
        if abs(planned - total_duration) > tolerance:
            raise ValueError(
                f"Planned total {planned}s misses target {total_duration:g}s "
                f"(tolerance {tolerance:g}s).")
    return {"clips": clips,
            "initial_state": _normalize_state_block(parsed.get("initial_state"))}


def plan_chain(user_prompt: str, *, model: str,
               base_url: str = "http://127.0.0.1:11434",
               num_clips: int | None = None,
               total_duration: float | None = None,
               timeout: int = 600,
               max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Text-only Qwen call: one prompt -> continuous action storyboard.

    Never sends images (no `images` key anywhere in the payload). Bounded
    retries on malformed/invalid JSON, then a clear error. Never fabricates.
    """
    constraints = []
    if num_clips is not None:
        constraints.append(f"Create EXACTLY {num_clips} clips.")
    else:
        constraints.append(
            f"Choose a sensible clip count ({MIN_AUTO_CLIPS}..{MAX_AUTO_CLIPS}; "
            f"no more than needed for this action).")
    if total_duration is not None:
        constraints.append(
            f"Clip durations must sum to approximately {total_duration:g} "
            f"seconds (each clip {MIN_CLIP_DURATION}..{MAX_CLIP_DURATION}s).")
    user_content = (
        "Plan a continuous video for this request. "
        + " ".join(constraints)
        + f"\n\nUser prompt: {user_prompt.strip()}\n"
          "Return ONLY the JSON object.")
    last_error: Exception | None = None
    for _ in range(1, max_attempts + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3},
        }
        try:
            result = ol._post(f"{base_url.rstrip('/')}/api/chat", payload,
                              timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Chain planning failed (model={model}): {exc}") from exc
        content = (result.get("message") or {}).get("content", "")
        if not content:
            last_error = ValueError("empty response")
            continue
        try:
            parsed = ol._extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        try:
            board = _validate_chain(parsed, num_clips, total_duration)
        except ValueError as exc:
            last_error = exc
            continue
        board["user_prompt"] = user_prompt.strip()
        return board
    raise RuntimeError(
        f"Chain planning failed after {max_attempts} attempts "
        f"(malformed output): {last_error}")


def select_clips(board: dict, num_clips: int | None) -> dict:
    """Limit a loaded board to the first N clips (--clips N).

    Returns the board unchanged when num_clips is None or already equal.
    Raises ValueError (never invents clips) when the board holds fewer.
    """
    total = len(board["clips"])
    if num_clips is None or num_clips == total:
        return board
    if num_clips > total:
        raise ValueError(
            f"Existing storyboard contains {total} clips, but --clips "
            f"{num_clips} was requested. Please edit the storyboard or "
            f"run with --clips {total}.")
    print(f"[Continuous] Storyboard has {total} clips; processing only "
          f"the first {num_clips} (--clips {num_clips}).")
    return {**board, "clips": board["clips"][:num_clips]}


def save_board(board: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(board, f, indent=2, ensure_ascii=False)


def load_board(path: str) -> dict:
    """Load + re-validate a persisted board. Corrupt files fail clearly."""
    if not os.path.isfile(path):
        raise ValueError(
            f"Continuous storyboard not found: {path}. Run the planning "
            f"command first (same --continuous command without --scene).")
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Continuous storyboard is corrupt ({path}): {exc}")
    try:
        board = _validate_chain(raw, None, None)
    except ValueError as exc:
        raise ValueError(f"Continuous storyboard is invalid: {exc}")
    board["user_prompt"] = str(raw.get("user_prompt", "") or "")
    board["source_image"] = str(raw.get("source_image", "") or "")
    board["planner_model"] = str(raw.get("planner_model", "") or "")
    return board


# -- identity (immutable master profile) -----------------------------------
#
# identity.json describes WHO/WHAT must stay the same for the whole chain.
# It is created once (never regenerated per clip) and its hash is recorded
# in the manifest: changing it marks every clip stale. Conservative by
# design: unknown fields stay empty, the source image remains authoritative.

IDENTITY_LOCK_LINES = [
    "same person throughout the entire sequence",
    "preserve facial identity",
    "preserve apparent age",
    "preserve hairstyle",
    "preserve clothing",
    "preserve body proportions",
    "preserve important accessories",
]

IDENTITY_SYSTEM_PROMPT = """You extract a CONSERVATIVE visual identity profile from TEXTUAL context only (a user prompt plus per-clip action summaries). You have NOT seen any image.
Describe only what the text clearly supports: identity central to every clip, apparent role, clothing, accessories, setting palette. Leave every uncertain field ("") empty - do NOT invent eye color, exact age, facial structure, or other specifics just to fill fields.

Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"identity": {"type": "...", "role": "...", "age_appearance": "...", "face": {"shape": "", "skin": "", "eyes": "", "nose": "", "facial_hair": ""}, "hair": "", "body": "", "clothing": "", "accessories": [], "color_palette": "", "visual_style": ""}}"""


def identity_path(out_dir: str) -> str:
    return os.path.join(out_dir, IDENTITY_FILENAME)


def memory_path(out_dir: str) -> str:
    return os.path.join(out_dir, MEMORY_FILENAME)


def default_identity() -> dict:
    """Empty (all-unknown) master identity. Never hallucinates."""
    return {
        "type": "none",
        "role": "",
        "age_appearance": "",
        "face": {"shape": "", "skin": "", "eyes": "", "nose": "",
                 "facial_hair": ""},
        "hair": "",
        "body": "",
        "clothing": "",
        "accessories": [],
        "color_palette": "",
        "visual_style": "",
    }


def normalize_identity(raw) -> dict:
    """Whitelist + coerce an identity profile. Unknown -> empty fields."""
    ident = default_identity()
    if not isinstance(raw, dict):
        return ident
    for key in ("type", "role", "age_appearance", "hair", "body",
                "clothing", "color_palette", "visual_style"):
        value = raw.get(key, "")
        if isinstance(value, str) and value.strip():
            ident[key] = value.strip()
    face = raw.get("face")
    if isinstance(face, dict):
        for key in ident["face"]:
            value = face.get(key, "")
            if isinstance(value, str) and value.strip():
                ident["face"][key] = value.strip()
    accessories = raw.get("accessories")
    if isinstance(accessories, list):
        ident["accessories"] = [str(a).strip() for a in accessories
                                if str(a).strip()]
    return ident


def save_identity(identity: dict, path: str,
                  characters: list | None = None) -> None:
    """Persist identity (+ optional character list).

    Shape is content-driven: no characters -> byte-compatible legacy v1
    document; with characters -> v2 document. Old projects are never
    churned by new code paths.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc = {"version": 1,
           "identity": normalize_identity(identity),
           "identity_lock": list(IDENTITY_LOCK_LINES)}
    if characters:
        chars = []
        for entry in characters:
            if not isinstance(entry, dict) or not entry.get("id"):
                raise ValueError(
                    "Character entry needs an 'id'.")
            chars.append({
                "id": str(entry["id"]),
                "reference_image": str(entry.get("reference_image", "")),
                "reference_sha256": str(entry.get("reference_sha256", "")),
                "identity": normalize_identity(entry.get("identity")),
                "identity_lock": list(IDENTITY_LOCK_LINES),
            })
        doc = {"characters": chars, **doc,
               "version": IDENTITY_VERSION}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def load_identity_doc(path: str) -> dict:
    """Full identity document: single profile + character list.

    v1 files (no 'characters') load fine with characters == []. Missing
    file -> FileNotFoundError. Anything malformed fails clearly.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Identity file is invalid (not an object): {path}")
    if not isinstance(raw.get("identity"), dict):
        raise ValueError(f"Identity file is invalid (no 'identity' object): "
                         f"{path}")
    characters = []
    raw_chars = raw.get("characters", [])
    if raw_chars is None:
        raw_chars = []
    if not isinstance(raw_chars, list):
        raise ValueError(f"Identity file has non-list 'characters': {path}")
    for entry in raw_chars:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ValueError(
                f"Identity file has a character entry without 'id': {path}")
        characters.append({
            "id": str(entry["id"]),
            "reference_image": str(entry.get("reference_image", "")),
            "reference_sha256": str(entry.get("reference_sha256", "")),
            "identity": normalize_identity(entry.get("identity")),
            "identity_lock": list(IDENTITY_LOCK_LINES),
        })
    return {"identity": normalize_identity(raw["identity"]),
            "identity_lock": list(IDENTITY_LOCK_LINES),
            "characters": characters}


def load_identity(path: str) -> dict:
    """Load a persisted identity. Missing file -> FileNotFoundError (the
    caller decides: one-time creation vs clear failure). Corrupt content
    always fails clearly - never guess an identity."""
    return load_identity_doc(path)["identity"]


def identity_hash(identity: dict) -> str:
    """Deterministic hash of the normalized profile (stale detection)."""
    canonical = json.dumps(normalize_identity(identity), sort_keys=True,
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- project-local character references --------------------------------------
#
# <project_dir>/ImageCharacterReference/ holds per-character visual
# anchors (Character_001.png, ...). The directory is OPTIONAL and strictly
# project-local: no global/shared directory is ever searched. Discovery is
# automatic (no CLI flag); the reference image is the visual source of
# truth while semantic fields stay conservative supporting metadata.
# identity.json v2 stores {"characters": [...]} alongside the unchanged
# legacy single profile; v1 files load fine (characters == []).

def character_ref_dir(base_dir: str) -> str:
    """Project-local reference directory. Never global."""
    return os.path.join(base_dir, CHARACTER_REF_DIRNAME)


def discover_character_refs(base_dir: str) -> list[dict]:
    """Find Character_<number>.<ext> files, natural-sorted by number.

    Returns [{"id", "reference_image" (project-relative, / separators),
    "abspath"}]. Missing/empty directory or no valid files -> [] (never
    fails: references are optional). Unrelated files (notes.txt,
    random.png, wrong extensions) are ignored.
    """
    ref_dir = character_ref_dir(base_dir)
    if not os.path.isdir(ref_dir):
        return []
    found = []
    for name in os.listdir(ref_dir):
        path = os.path.join(ref_dir, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in CHARACTER_EXTENSIONS:
            continue
        match = CHARACTER_ID_RE.match(stem)
        if not match:
            continue
        found.append({
            "id": f"Character_{match.group(1)}",
            "reference_image": f"{CHARACTER_REF_DIRNAME}/{name}",
            "abspath": os.path.abspath(path),
            "_num": int(match.group(1)),
        })
    found.sort(key=lambda e: (e["_num"], e["reference_image"].lower()))
    seen: dict[str, str] = {}
    for entry in found:
        if entry["id"] in seen:
            raise ValueError(
                f"Duplicate character id {entry['id']!r}: "
                f"{seen[entry['id']]} and {entry['reference_image']} "
                f"map to the same id. Keep one file per character.")
        seen[entry["id"]] = entry["reference_image"]
        del entry["_num"]
    return found


def build_character_entry(discovered: dict,
                          existing: dict | None = None) -> dict:
    """Merge one discovered file with its persisted entry (if any).

    Semantic profiles are immutable: a surviving id keeps its profile, a
    new id starts from the conservative default (the image itself is the
    source of truth; text-only Qwen never inspects it). reference_sha256
    is always recomputed fresh so file swaps are detectable.
    """
    sha = file_hash(discovered["abspath"])
    if sha is None:
        raise ValueError(
            f"Character reference unreadable: {discovered['abspath']} "
            f"(id {discovered['id']}).")
    identity = default_identity()
    if isinstance(existing, dict):
        identity = normalize_identity(existing.get("identity"))
    return {"id": discovered["id"],
            "reference_image": discovered["reference_image"],
            "reference_sha256": sha,
            "identity": identity,
            "identity_lock": list(IDENTITY_LOCK_LINES)}


def merge_character_refs(existing: list,
                         discovered: list[dict]) -> list[dict]:
    """Reconcile persisted entries with current discovery.

    Surviving ids keep their semantic profiles; new ids start default;
    removed files drop out (their scenes then fail loudly at resolve
    time instead of using a ghost reference).
    """
    by_id = {}
    for entry in existing or []:
        if isinstance(entry, dict) and entry.get("id"):
            by_id[str(entry["id"])] = entry
    return [build_character_entry(d, by_id.get(d["id"])) for d in discovered]


def characters_hash(characters: list) -> str:
    """Deterministic hash over id + project-relative path + content sha +
    normalized semantic profile. Absolute paths never enter the hash
    (projects stay portable). Unrelated filesystem changes can't move it."""
    canonical = json.dumps(
        [{"id": str(c.get("id", "")),
          "reference_image": str(c.get("reference_image", "")),
          "reference_sha256": str(c.get("reference_sha256", "")),
          "identity": normalize_identity(c.get("identity"))}
         for c in (characters or []) if isinstance(c, dict)],
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_chain_characters(base_dir: str) -> list[dict]:
    """Characters for a run: identity.json mapping + fresh on-disk shas.

    Missing identity.json -> [] (legacy projects). Corrupt identity.json
    or a missing/unreadable reference file fails clearly.
    """
    path = identity_path(os.path.join(base_dir, CLIPS_DIRNAME))
    if not os.path.isfile(path):
        return []
    doc = load_identity_doc(path)
    resolved = []
    for entry in doc["characters"]:
        abspath = os.path.normpath(os.path.join(
            base_dir, *str(entry["reference_image"]).split("/")))
        sha = file_hash(abspath)
        if sha is None or not os.path.isfile(abspath):
            raise ValueError(
                f"Character reference missing or unreadable: "
                f"{entry['reference_image']} (id {entry['id']}). Restore "
                f"the file or remove the id from scenes using it.")
        resolved.append({**entry, "abspath": abspath,
                         "reference_sha256": sha})
    return resolved


def resolve_clip_characters(clip: dict,
                            characters: list[dict]) -> list[dict]:
    """Entries for the ids a scene lists. Unknown ids fail clearly."""
    wanted = clip.get("characters") or []
    if not wanted:
        return []
    by_id = {c["id"]: c for c in characters}
    resolved = []
    for cid in wanted:
        if cid not in by_id:
            raise ValueError(
                f"Clip {clip.get('id', '?')} references unknown character "
                f"{cid!r} (no matching file in "
                f"{CHARACTER_REF_DIRNAME}/).")
        resolved.append(by_id[cid])
    return resolved


def create_identity_profile(user_prompt: str, clips: list[dict], *,
                            model: str,
                            base_url: str = "http://127.0.0.1:11434",
                            timeout: int = 600,
                            max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Text-only Qwen call: prompt + clip summaries -> identity profile.

    Never sends images. Conservative by instruction + normalization:
    anything uncertain comes back empty.
    """
    summary = "\n".join(
        f"Clip {c['id']}: {c.get('action', '')} "
        f"(ends: {c.get('end_state', '')})" for c in clips)
    user_content = (
        "Describe the single central identity that must stay visually "
        "consistent across ALL of the following clips. Return ONLY the "
        "JSON object.\n\n"
        f"User prompt: {user_prompt.strip()}\n\n"
        f"Clip summaries:\n{summary}")
    last_error: Exception | None = None
    for _ in range(1, max_attempts + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": IDENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            result = ol._post(f"{base_url.rstrip('/')}/api/chat", payload,
                              timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Identity creation failed (model={model}): {exc}") from exc
        content = (result.get("message") or {}).get("content", "")
        if not content:
            last_error = ValueError("empty response")
            continue
        try:
            parsed = ol._extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict) \
                or not isinstance(parsed.get("identity"), dict):
            last_error = ValueError("response has no 'identity' object")
            continue
        return normalize_identity(parsed["identity"])
    raise RuntimeError(
        f"Identity creation failed after {max_attempts} attempts "
        f"(malformed output): {last_error}")


def render_identity_block(identity: dict) -> str:
    """Compact identity lock: fixed lines + only non-empty profile facts."""
    ident = normalize_identity(identity)
    lines = ["IDENTITY LOCK:"] + [f"- {line}" for line in IDENTITY_LOCK_LINES]
    facts = []
    for key in ("type", "role", "age_appearance", "hair", "body",
                "clothing", "color_palette", "visual_style"):
        if ident.get(key) and ident[key].lower() != "none":
            facts.append(f"{key.replace('_', ' ')}: {ident[key]}")
    face = " ".join(f"{k}: {v}" for k, v in ident["face"].items() if v)
    if face:
        facts.append(f"face ({face})")
    if ident["accessories"]:
        facts.append("accessories: " + ", ".join(ident["accessories"]))
    if facts:
        lines.append("Identity: " + "; ".join(facts) + ".")
    return "\n".join(lines)


# -- continuity memory (mutable runtime state) -------------------------------
#
# memory.json describes WHAT HAS HAPPENED and WHAT STATE the story is in.
# Snapshots are pure functions of (board, completed_clip_count), so a
# missing/corrupt memory file is always safely reconstructible and resume
# never needs Qwen.

STATE_KEYS = ("global", "characters", "objects", "camera")


def _normalize_str_dict(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in raw.items()
            if str(v).strip()}


def _normalize_entry_list(raw) -> list:
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if isinstance(entry, dict):
            norm = {str(k).strip(): str(v).strip() for k, v in entry.items()
                    if str(v).strip()}
            if norm:
                out.append(norm)
        elif str(entry).strip():
            out.append({"entry": str(entry).strip()})
    return out


def _normalize_state_block(raw) -> dict:
    """Tolerant gate for initial_state / per-clip state blocks."""
    block = {"global": {}, "characters": [], "objects": [], "camera": {}}
    if not isinstance(raw, dict):
        return block
    block["global"] = _normalize_str_dict(raw.get("global"))
    block["camera"] = _normalize_str_dict(raw.get("camera"))
    block["characters"] = _normalize_entry_list(raw.get("characters"))
    block["objects"] = _normalize_entry_list(raw.get("objects"))
    return block


def blank_memory() -> dict:
    return {"version": MEMORY_VERSION, "global": {}, "characters": [],
            "objects": [], "camera": {},
            "story_state": {"completed": [], "current": "", "next": ""},
            "last_completed_clip": 0}


def normalize_memory(raw) -> dict:
    """Tolerant gate for a persisted memory file."""
    mem = blank_memory()
    if not isinstance(raw, dict):
        return mem
    block = _normalize_state_block(raw)
    mem["global"] = block["global"]
    mem["characters"] = block["characters"]
    mem["objects"] = block["objects"]
    mem["camera"] = block["camera"]
    story = raw.get("story_state")
    if isinstance(story, dict):
        completed = story.get("completed")
        mem["story_state"] = {
            "completed": [str(c).strip() for c in completed
                          if str(c).strip()] if isinstance(completed, list)
            else [],
            "current": str(story.get("current", "") or "").strip(),
            "next": str(story.get("next", "") or "").strip(),
        }
    try:
        mem["last_completed_clip"] = int(raw.get("last_completed_clip", 0))
    except (TypeError, ValueError):
        mem["last_completed_clip"] = 0
    return mem


def save_memory(memory: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(normalize_memory(memory), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_memory(path: str) -> dict | None:
    """Persisted memory or None when missing/corrupt (reconstructible)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return normalize_memory(raw)


def initial_memory(board: dict) -> dict:
    """Memory before clip 1: planner initial state + opening story state."""
    base = blank_memory()
    block = _normalize_state_block(board.get("initial_state"))
    base["global"] = block["global"]
    base["characters"] = block["characters"]
    base["objects"] = block["objects"]
    base["camera"] = block["camera"]
    return snapshot_after(board, base, 0)


def snapshot_after(board: dict, initial: dict, completed_id: int) -> dict:
    """Deterministic memory after clip N: cumulative per-clip states folded
    over the initial snapshot, story advanced. Pure function of the board."""
    mem = json.loads(json.dumps(initial))  # deep copy via JSON
    clips = board.get("clips", [])
    for clip in clips:
        if int(clip.get("id", 0)) > completed_id:
            break
        state = clip.get("state") or {}
        if not isinstance(state, dict):
            continue
        block = _normalize_state_block(state)
        if block["global"]:
            mem["global"] = {**mem.get("global", {}), **block["global"]}
        if block["camera"]:
            mem["camera"] = {**mem.get("camera", {}), **block["camera"]}
        if block["characters"]:
            mem["characters"] = block["characters"]
        if block["objects"]:
            mem["objects"] = block["objects"]
    mem["story_state"] = {
        "completed": [str(c.get("end_state", "") or "").strip()
                      for c in clips if int(c.get("id", 0)) <= completed_id],
        "current": str(clips[completed_id].get("action", "") or "").strip()
        if 0 <= completed_id < len(clips) else (
            str(clips[-1].get("end_state", "") or "").strip()
            if clips and completed_id >= len(clips) else ""),
        "next": str(clips[completed_id + 1].get("action", "") or "").strip()
        if 0 <= completed_id + 1 < len(clips) else "",
    }
    mem["last_completed_clip"] = completed_id
    return normalize_memory(mem)


def memory_hash(memory: dict) -> str:
    canonical = json.dumps(normalize_memory(memory), sort_keys=True,
                           ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reconstruct_memory(board: dict, base_mem: dict, manifest: dict) -> dict:
    """Rebuild memory from board + manifest completion (no Qwen needed).

    Used when memory.json is missing/corrupt: completion is taken from
    manifest entries whose recorded outputs still exist on disk.
    """
    done = [int(k) for k, v in manifest.items()
            if isinstance(v, dict) and isinstance(v.get("video"), str)
            and isinstance(v.get("lastframe"), str)
            and os.path.isfile(v["video"])
            and os.path.isfile(v["lastframe"])]
    return snapshot_after(board, base_mem, max(done) if done else 0)


def _render_entries(entries: list) -> str:
    parts = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        head = entry.get("id") or entry.get("role") or entry.get("entry") \
            or "item"
        rest = ", ".join(f"{k}: {v}" for k, v in entry.items()
                         if k not in ("id",) and v)
        parts.append(f"{head} ({rest})" if rest else str(head))
    return " | ".join(parts)


def render_memory_block(memory: dict, clip_id: int) -> str:
    """Compact continuity section for one clip's prompt."""
    mem = normalize_memory(memory)
    lines = [f"CONTINUITY STATE (after clip {clip_id - 1}):"
             if clip_id > 1 else "INITIAL STATE:"]
    if mem["global"]:
        lines.append("Setting: " + "; ".join(
            f"{k}: {v}" for k, v in mem["global"].items()) + ".")
    if mem["characters"]:
        lines.append("Characters: " + _render_entries(mem["characters"]) + ".")
    if mem["objects"]:
        lines.append("Objects: " + _render_entries(mem["objects"]) + ".")
    if mem["camera"]:
        lines.append("Camera: " + "; ".join(
            f"{k}: {v}" for k, v in mem["camera"].items()) + ".")
    story = mem["story_state"]
    done = [s for s in story["completed"] if s]
    if done:
        lines.append("Already happened: " + " / ".join(done[-2:]) + ".")
    if story["current"]:
        lines.append("Now: " + story["current"])
    if story["next"]:
        lines.append("Next: " + story["next"])
    return "\n".join(lines)


def ensure_identity(out_dir: str, user_prompt: str, board: dict, *,
                    model: str, base_url: str = "http://127.0.0.1:11434",
                    timeout: int = 600,
                    identity_file: str | None = None,
                    base_dir: str | None = None) -> dict:
    """Load or one-time-create the immutable master identity.

    Precedence: explicit --identity-file > existing identity.json >
    one-time Qwen creation (also used as a loud one-time upgrade when
    clips already exist but predate identity tracking). A corrupt
    identity.json always fails clearly instead of being guessed.
    Changing the identity later marks every clip stale via manifest
    identity hashes (never mixes identities silently).

    When base_dir is given, project-local character references
    (<base_dir>/ImageCharacterReference/) are discovered and merged into
    identity.json (new ids start from the conservative default profile;
    surviving ids keep theirs). Returns the single master profile, as
    before; per-character entries travel via load_chain_characters().
    """
    path = identity_path(out_dir)
    existing_doc = None
    if identity_file:
        if not os.path.isfile(identity_file):
            raise FileNotFoundError(
                f"Identity file not found: {identity_file}")
        try:
            with open(identity_file, encoding="utf-8") as f:
                raw = json.load(f)
            identity = normalize_identity(
                raw.get("identity", raw) if isinstance(raw, dict) else None)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Identity file is invalid ({identity_file}): {exc}")
        print(f"  Using explicit identity profile ({identity_file}).")
    elif os.path.isfile(path):
        try:
            doc = load_identity_doc(path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Identity file is corrupt ({path}): {exc}. Refusing to "
                f"guess an identity; fix or delete the file to recreate it.")
        identity = doc["identity"]
        existing_doc = doc
        print(f"  Loaded master identity (immutable).")
    else:
        print("  Creating master identity profile (one-time, immutable)...")
        identity = create_identity_profile(
            user_prompt, board.get("clips", []), model=model,
            base_url=base_url, timeout=timeout)
        print(f"  Saved master identity -> {path}")
    characters = None
    if base_dir is not None:
        discovered = discover_character_refs(base_dir)
        prior = existing_doc["characters"] if existing_doc else []
        characters = merge_character_refs(prior, discovered) or None
        if characters:
            print(f"  Character references: "
                  f"{', '.join(c['id'] for c in characters)} "
                  f"({len(characters)} file(s) in "
                  f"{CHARACTER_REF_DIRNAME}/)")
    save_identity(identity, path, characters)
    return identity


def _render_character_line(entry: dict) -> str:
    """One compact line per referenced character: id + known facts only."""
    ident = normalize_identity(entry.get("identity"))
    facts = [p for p in (ident.get("role", ""), ident.get("clothing", ""))
             if p]
    for key in ("hair", "age_appearance", "body"):
        if ident.get(key):
            facts.append(f"{key}: {ident[key]}")
    if ident.get("accessories"):
        facts.append("with " + ", ".join(ident["accessories"]))
    detail = "; ".join(facts) if facts else "visual reference on file"
    return f"- {entry['id']}: {detail}"


def build_video_prompt(clip: dict, identity: dict | None = None,
                       mem_snapshot: dict | None = None,
                       characters: list | None = None) -> str:
    """Final backend prompt.

    Without identity/memory this is exactly the legacy behavior (bare
    action prompt for clip 1, CONTINUITY_BLOCK prefix after). With them,
    the prompt is composed from Identity Lock + Character Reference
    Context + Continuity Memory + Current Action + End State + Camera +
    Continuation. Only characters listed by THIS clip appear (never the
    whole cast), keeping prompts compact.
    """
    action_prompt = clip["video_prompt"].strip()
    if identity is None:
        if int(clip["id"]) <= 1:
            return action_prompt
        return CONTINUITY_BLOCK + action_prompt
    mem = normalize_memory(mem_snapshot or {})
    camera = "; ".join(f"{k}: {v}" for k, v in mem["camera"].items())
    parts = [
        render_identity_block(identity),
    ]
    if characters:
        parts.append("CHARACTER REFERENCE:\n" + "\n".join(
            _render_character_line(c) for c in characters))
    parts += [
        render_memory_block(mem, int(clip["id"])),
        "CURRENT ACTION:\n" + clip["action"].strip(),
        "END STATE:\n" + clip["end_state"].strip(),
        "CAMERA:\n" + (camera + "." if camera else
                       "Maintain the previous camera; move only if the "
                       "action requires it."),
    ]
    if int(clip["id"]) > 1:
        parts.append(
            "CONTINUATION:\n"
            "Continue directly from the provided starting frame. "
            "Do not restart the action. Do not reset the environment. "
            "Do not change the character identity.")
    return "\n\n".join(parts)


# -- engine ------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _tmp(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.tmp{ext}"


def _cleanup_tmp(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _copy_start(src: str, dest: str) -> None:
    """Materialize a clip's start frame (copy, never move/symlink)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    shutil.copyfile(src, dest)


def ensure_start_frame(out_dir: str, clip_id: int, prev_lastframe: str | None,
                       source_image: str) -> str:
    """Resolve + materialize clip N's start image. Clear errors, no guessing.

    Clip 1 starts from the source image. Later clips REQUIRE the previous
    clip's last frame; falling back to the source image is refused (it
    would silently restart the action).
    """
    dest = start_path(out_dir, clip_id)
    if clip_id <= 1:
        if not st.valid_image_file(source_image):
            raise ValueError(
                f"Source image missing or invalid: {source_image}.")
        _copy_start(source_image, dest)
        return dest
    if prev_lastframe is None or not st.valid_image_file(prev_lastframe):
        raise ValueError(
            f"Cannot start clip {clip_id}: previous last frame missing or "
            f"invalid ({prev_lastframe or 'none'}). Generate clip "
            f"{clip_id - 1} first; refusing to restart from the source "
            f"image.")
    if os.path.isfile(dest) and file_hash(dest) == file_hash(prev_lastframe):
        return dest
    _copy_start(prev_lastframe, dest)
    return dest


def _expected_duration(provider, clip: dict) -> float | None:
    """Planned duration, or None for fixed-length backends (wan_animate2).

    Fixed backends declare `fixed_duration`; their outputs are validated by
    existence/readability instead of storyboard duration (same convention as
    run_video_stage in main.py).
    """
    if getattr(provider, "fixed_duration", False):
        return None
    return float(clip["duration"])


def clip_outputs_valid(out_dir: str, clip: dict, provider=None) -> bool:
    """Filesystem truth for one clip: video ~planned duration + real image."""
    video = video_path(out_dir, int(clip["id"]))
    lastframe = lastframe_path(out_dir, int(clip["id"]))
    return (st.valid_video_file(video, _expected_duration(provider, clip))
            and st.valid_image_file(lastframe))


def run_chain(*, out_dir: str, board: dict, source_image: str,
              provider, negative_prompt: str, width: int, height: int,
              seed: int, ffmpeg_exe: str, only_clip: int | None = None,
              motion_clips: list | None = None,
              identity: dict | None = None,
              initial_mem: dict | None = None) -> dict:
    """Execute clips sequentially with last-frame chaining.

    Returns {clip_id: video_path}. Skips clips whose outputs are valid AND
    whose recorded start-frame hash still matches (resume). A regenerated
    clip changes its last frame, so downstream clips automatically become
    stale and regenerate (cascade). `only_clip` regenerates exactly one
    clip and warns that downstream clips may now be stale.

    With `identity` supplied, prompts are composed from Identity Lock +
    Continuity Memory + action, memory.json is saved after every clip, and
    the manifest additionally records identity/memory hashes: changing the
    identity (or the board state feeding a clip) marks downstream clips
    stale. Entries lacking hashes (pre-upgrade runs) are grandfathered so
    old resumes keep working. Without identity the legacy prompt format
    and manifest shape are used unchanged.
    """
    clips = board["clips"]
    ids = [int(c["id"]) for c in clips]
    if only_clip is not None:
        if only_clip not in ids:
            raise ValueError(
                f"--scene {only_clip} out of range "
                f"(chain has clips {ids[0]}..{ids[-1]}).")
        todo = [only_clip]
    else:
        todo = ids
    if motion_clips is not None and len(motion_clips) < max(todo):
        raise ValueError(
            f"Only {len(motion_clips)} motion clip(s) available for "
            f"{max(todo)} chained clip(s).")

    manifest = load_manifest(os.path.join(out_dir, MANIFEST_FILENAME))
    if not isinstance(manifest, dict):
        manifest = {}
    results: dict[int, str] = {}

    use_lock = identity is not None
    ihash = identity_hash(identity) if use_lock else None
    base_mem = initial_mem if initial_mem is not None \
        else initial_memory(board)
    # Project-local character references (harmless [] when absent).
    # Resolved once here so every clip sees the same mapping; staleness
    # is tracked per run via characters_hash in the manifest.
    base_dir = os.path.dirname(os.path.abspath(out_dir))
    try:
        chain_chars = load_chain_characters(base_dir)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Character references unusable: {exc}") from exc
    chash = characters_hash(chain_chars)
    if chain_chars:
        print(f"  Character references: "
              f"{', '.join(c['id'] for c in chain_chars)}")
    mem_file = os.path.join(out_dir, MEMORY_FILENAME)
    if use_lock:
        loaded = load_memory(mem_file)
        if loaded is None:
            print("  (memory.json missing or unreadable; reconstructing "
                  "deterministically from the storyboard)")
            save_memory(_reconstruct_memory(board, base_mem, manifest),
                        mem_file)
        else:
            print(f"  Loaded continuity memory (state after clip "
                  f"{loaded.get('last_completed_clip', 0)}).")

    if only_clip is not None and only_clip < max(ids):
        print(f"  WARNING: regenerating only clip {only_clip}; downstream "
              f"clips {only_clip + 1}..{max(ids)} start from its last frame "
              f"and may require regeneration.")

    for clip in clips:
        cid = int(clip["id"])
        if cid not in todo and only_clip is not None:
            continue
        prev_last = lastframe_path(out_dir, cid - 1) if cid > 1 else None
        start = ensure_start_frame(out_dir, cid, prev_last, source_image)
        start_hash = file_hash(start)
        entry = manifest.get(str(cid), {})
        video = video_path(out_dir, cid)
        lastframe = lastframe_path(out_dir, cid)
        mem_in = snapshot_after(board, base_mem, cid - 1) \
            if use_lock else None
        mem_out_hash = memory_hash(snapshot_after(board, base_mem, cid)) \
            if use_lock else None
        stale_hashes = False
        if use_lock and isinstance(entry, dict):
            if "identity_hash" in entry and entry.get("identity_hash") != ihash:
                stale_hashes = True
            if "memory_hash" in entry and entry.get("memory_hash") != mem_out_hash:
                stale_hashes = True
            # Absent characters_hash = pre-upgrade entry: grandfathered so
            # old resumes keep working; present-but-different = stale.
            if "characters_hash" in entry \
                    and entry.get("characters_hash") != chash:
                stale_hashes = True
        fresh_start_needed = (
            not clip_outputs_valid(out_dir, clip, provider)
            or not isinstance(entry, dict)
            or start_hash is None
            or entry.get("start_frame_hash") != start_hash
            or stale_hashes
        )
        if cid not in todo:
            # Cascade check outside --scene scope: a stale downstream clip
            # (start frame changed) must regenerate to stay continuous.
            if not fresh_start_needed:
                results[cid] = video
                continue
            print(f"  Clip {cid:03d} start frame changed upstream; "
                  f"regenerating to stay continuous.")
        elif only_clip is None and not fresh_start_needed:
            print(f"  Clip {cid:03d}: reused "
                  f"({os.path.basename(video)}, valid)")
            results[cid] = video
            continue
        elif only_clip is not None and not fresh_start_needed:
            print(f"  Clip {cid:03d}: forced regeneration (--scene).")

        motion = None
        if motion_clips is not None:
            motion = motion_clips[cid - 1]
        try:
            clip_chars = resolve_clip_characters(clip, chain_chars)
        except ValueError as exc:
            raise ValueError(f"Character resolution failed: {exc}") from exc
        prompt = build_video_prompt(clip, identity, mem_in, clip_chars)
        with open(prompt_path(out_dir, cid), "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"  Clip {cid:03d}/{len(clips)}: generating "
              f"({clip['duration']}s) ...")
        tmp = _tmp(video)
        try:
            kwargs = {}
            if motion is not None:
                kwargs["motion_video"] = _Path(motion)
            # identity_reference: this scene's first character ref when the
            # scene names characters, else the master source image. No
            # backend consumes it yet (LTX/Wan read input_image only); the
            # hook exists for future identity-aware backends.
            ref_path = clip_chars[0]["abspath"] if clip_chars \
                else source_image
            provider.generate(VideoRequest(
                prompt=prompt, negative_prompt=negative_prompt,
                input_image=start, width=width, height=height,
                duration=float(clip["duration"]), seed=seed + 1000 + cid,
                output_path=tmp,
                identity_reference=_Path(ref_path), **kwargs))
        except KeyboardInterrupt:
            _cleanup_tmp(tmp)
            raise
        except Exception:
            _cleanup_tmp(tmp)
            raise
        if not st.valid_video_file(tmp, _expected_duration(provider, clip)):
            _cleanup_tmp(tmp)
            raise RuntimeError(
                f"Clip {cid:03d} output failed validation "
                f"(expected ~{clip['duration']}s).")
        os.replace(tmp, video)
        print(f"  Extracting last frame of clip {cid:03d} ...")
        try:
            ff.extract_last_frame(video, lastframe, executable=ffmpeg_exe)
        except Exception as exc:
            raise RuntimeError(
                f"Last-frame extraction failed for clip {cid:03d}: "
                f"{exc}") from exc
        if not st.valid_image_file(lastframe):
            raise RuntimeError(
                f"Last frame of clip {cid:03d} failed validation.")
        manifest[str(cid)] = {
            "video": video,
            "lastframe": lastframe,
            "start_frame": start,
            "start_frame_hash": file_hash(start),
        }
        if use_lock:
            manifest[str(cid)]["identity_hash"] = ihash
            manifest[str(cid)]["memory_hash"] = mem_out_hash
            manifest[str(cid)]["characters_hash"] = chash
            save_memory(snapshot_after(board, base_mem, cid), mem_file)
        save_manifest(os.path.join(out_dir, MANIFEST_FILENAME), manifest)
        print(f"    -> {video}")
        results[cid] = video
    return results


def concat_chain(out_dir: str, clips: list[dict], ffmpeg_exe: str) -> str:
    """Concatenate all clip videos in numeric order -> continuous/final.mp4."""
    ordered = [video_path(out_dir, int(c["id"])) for c in clips]
    missing = [p for p in ordered if not st.valid_video_file(p)]
    if missing:
        raise ValueError(
            "Cannot concatenate: missing or invalid clip(s): "
            + ", ".join(os.path.basename(os.path.dirname(p)) for p in missing)
            + ". Generate them first.")
    final = os.path.join(out_dir, FINAL_FILENAME)
    tmp = _tmp(final)
    ff.concat_scenes(ordered, tmp, executable=ffmpeg_exe)
    if not st.valid_video_file(tmp):
        _cleanup_tmp(tmp)
        raise RuntimeError("Concatenated chain output failed validation.")
    os.replace(tmp, final)
    print(f"Done -> {final}")
    return final
