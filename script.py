"""Mode C: long source text -> Qwen condensation -> video script -> storyboard.

Two-pass director (text-only; never sends image payloads, never needs
vision): PASS 1 understands/condenses the source into a persisted video
script; PASS 2 converts that script plus ONE master subject profile into
the existing storyboard schema. Reuses the Ollama transport, JSON
extraction, model-availability check (ollama.py) and the prompt-leak
guard (subject.py). Kept separate from Mode B (keyframes -> Qwen-VL).
"""
from __future__ import annotations

import json
import os
import re

import ollama as ol

# Single-request ceiling: at/below this the source is understood + condensed
# in one Qwen call (logical passes preserved, calls collapsed).
SINGLE_LIMIT_CHARS = 6000
# Chunk target for long sources; sentence boundaries are never cut.
CHUNK_SIZE_CHARS = 4000
# Hard stop against runaway bills/time on gigantic inputs.
MAX_CHUNKS = 60
# Per-request attempts for malformed JSON / invalid content.
MAX_ATTEMPTS = 3
# Mode C scene durations (spec: 5..10s; storyboard duration wins downstream).
MIN_DURATION = 5
MAX_DURATION = 10
DEFAULT_DURATION = 7

# Master subject presets. "auto" (default) lets Qwen derive the profile from
# the source; presets force one. Extend this dict for future subjects.
SUBJECT_PRESETS = {
    "none": {"type": "none", "identity": "", "features": "",
             "clothing_or_equipment": "", "consistency": ""},
    "god": {"type": "character",
            "identity": "God",
            "features": ("serene, wise and compassionate face with gentle "
                         "eyes, long hair and a full beard"),
            "clothing_or_equipment": "flowing white robes with gold trim",
            "consistency": ("the same face, hair, beard, white-and-gold "
                            "robes and warm radiant style in every scene")},
}

CONDENSE_SYSTEM = """You are a script director condensing a long source text into a concise video-oriented script.
Read and understand the provided source carefully. Condense it specifically for a visual storytelling video. Preserve the central meaning, important events, characters, locations, chronology, and theological/contextual meaning of the source. Remove repetition and details that do not contribute to a visual video. Do not invent major events, characters, locations, dialogue, miracles, actions, or outcomes that are not supported by the source. You may reorganize information slightly for clear visual storytelling, but do not change the meaning of the source. The result must be concise enough to become a video script.

Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"title": "...", "summary": "...", "key_events": ["..."], "characters": ["..."], "locations": ["..."], "video_script": [{"narration": "...", "visual": "...", "source_reference": "..."}]}

Rules:
- video_script entries are ordered as they should appear in the video; narration is clean spoken text (no JSON, stage directions, camera instructions, or meta commentary); visual is a short visual note for the scene planner; source_reference names the source part it comes from.
- key_events/characters/locations list only what the source supports."""

EXTRACT_SYSTEM = """You extract story structure from ONE chunk of a longer source text for a video adaptation.
Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"key_events": ["..."], "characters": ["..."], "locations": ["..."], "notes": "..."}

Rules:
- List only events, characters and locations stated or clearly implied in THIS chunk, in the order they appear. Do not invent anything from other chunks. notes is one sentence of context."""


def _storyboard_system(num_scenes: int, subject: dict) -> str:
    subject_json = json.dumps(subject, ensure_ascii=False)
    return f"""You are a storyboard director turning a condensed video script into exactly {num_scenes} storyboard scenes.
The source controls the story: creative visualization (camera angle, lighting, atmosphere, composition, cinematic framing, reasonable non-contradictory environmental detail) is allowed; inventing major events, new characters, new miracles, new locations, new dialogue presented as scripture, changed outcomes, or changed meaning is NOT allowed. Narration must come from the condensed script, kept faithful in wording and meaning, as clean spoken text (no JSON, stage directions, camera instructions, excessive punctuation, or meta commentary) suitable for text-to-speech.

One master subject profile governs every scene (it is applied automatically; do not restate the full identity inside every image prompt, and never reinvent it):
{subject_json}

Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{{"subject": {{"type": "...", "identity": "...", "features": "...", "clothing_or_equipment": "...", "consistency": "..."}}, "scenes": [{{"id": 1, "duration": 7, "image_prompt": "...", "video_prompt": "...", "narration": "..."}}]}}

Rules:
- Exactly {num_scenes} scenes, ids 1..{num_scenes} in order.
- duration is an integer 5..10 seconds, chosen from narration length and visual complexity (vary it; do not hard-code every scene).
- image_prompt: pure visual prose for image generation (environment, characters, objects, composition, camera viewpoint, lighting, time of day, atmosphere, visible action, cinematic framing) consistent with the scene's visual note. No schema labels, JSON keys, scene IDs, or instructions.
- video_prompt: plausible motion for the image (camera move plus natural subject/environment motion); must not contradict the image_prompt.
- narration: spoken narration for the scene from the condensed script.
- subject: echo the master profile above (type/identity/features/clothing_or_equipment/consistency)."""


def load_source_text(path: str) -> str:
    """Read a source text file. Clear errors, never empty downstream."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Script file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise ValueError(f"Cannot read script file {path}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"Script file is empty: {path}")
    return text


def normalize_text(text: str) -> str:
    """Collapse whitespace; preserve sentence order."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Split on sentence boundaries so no sentence is cut when avoidable."""
    sentences = [s.strip() for s in
                 re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # One pathological sentence: hard-split by characters.
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _chat_json(messages: list[dict], *, model: str, base_url: str,
               timeout: int, what: str,
               max_attempts: int = MAX_ATTEMPTS) -> dict:
    """One text-only chat call with bounded retry on malformed JSON."""
    last_error: Exception | None = None
    for _ in range(1, max_attempts + 1):
        payload = {"model": model, "messages": messages, "stream": False,
                   "format": "json", "options": {"temperature": 0.2}}
        try:
            result = ol._post(f"{base_url.rstrip('/')}/api/chat", payload,
                              timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Script analysis failed ({what}, model={model}): "
                f"{exc}") from exc
        content = (result.get("message") or {}).get("content", "")
        if not content:
            last_error = ValueError("empty response")
            continue
        try:
            parsed = ol._extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError("response is not a JSON object")
            continue
        return parsed
    raise RuntimeError(
        f"Script analysis failed ({what}) after {max_attempts} attempts "
        f"(malformed output): {last_error}")


def _merge_extracts(extracts: list[dict]) -> dict:
    """Order-preserving union of per-chunk extracts."""
    merged: dict[str, list] = {"key_events": [], "characters": [],
                               "locations": [], "notes": []}

    def add(key: str, values) -> None:
        if not isinstance(values, list):
            return
        for value in values:
            text = str(value).strip()
            if text and text not in merged[key]:
                merged[key].append(text)

    for extract in extracts:
        if not isinstance(extract, dict):
            continue
        add("key_events", extract.get("key_events"))
        add("characters", extract.get("characters"))
        add("locations", extract.get("locations"))
        note = str(extract.get("notes", "") or "").strip()
        if note:
            merged["notes"].append(note)
    return merged


def condense_source(text: str, *, model: str,
                    base_url: str = "http://127.0.0.1:11434",
                    timeout: int = 600) -> dict:
    """PASS 1: source text -> condensed video-script dict (validated)."""
    normalized = normalize_text(text)
    if len(normalized) <= SINGLE_LIMIT_CHARS:
        print("Condensing source in a single request")
        messages = [
            {"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content":
             "Source text to condense for video:\n\n" + normalized},
        ]
        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            parsed = _chat_json(messages, model=model, base_url=base_url,
                                timeout=timeout, what="condensation")
            try:
                return _validate_condensed(parsed)
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            f"Script analysis failed (condensation) after {MAX_ATTEMPTS} "
            f"attempts: {last_error}")
    chunks = chunk_text(normalized)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"Source too long: {len(chunks)} chunks exceed the limit "
            f"of {MAX_CHUNKS}; shorten the source text.")
    print(f"Source too long for one request; analyzing "
          f"{len(chunks)} chunks")
    extracts = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [chunk {i}/{len(chunks)}] extracting structure...")
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content":
             f"Chunk {i} of {len(chunks)}:\n\n{chunk}"},
        ]
        extracts.append(_chat_json(
            messages, model=model, base_url=base_url, timeout=timeout,
            what=f"chunk {i}/{len(chunks)}"))
    merged = _merge_extracts(extracts)
    print("Condensing merged structure into the video script...")
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM},
        {"role": "user", "content":
         "Merged story structure extracted from a long source "
         "(in source order). Condense it into the video script:\n\n"
         + json.dumps(merged, ensure_ascii=False)},
    ]
    last_error = None
    for _ in range(MAX_ATTEMPTS):
        parsed = _chat_json(messages, model=model, base_url=base_url,
                            timeout=timeout, what="condensation")
        try:
            return _validate_condensed(parsed)
        except RuntimeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Script analysis failed (condensation) after {MAX_ATTEMPTS} "
        f"attempts: {last_error}")


def _validate_condensed(parsed: dict) -> dict:
    """Structural gate for the condensed script. Never fabricates."""
    if not isinstance(parsed, dict):
        raise RuntimeError("Condensation did not return a JSON object.")
    script = parsed.get("video_script")
    if not isinstance(script, list) or not script:
        raise RuntimeError(
            "Condensation returned no video_script entries.")
    for entry in script:
        if not isinstance(entry, dict):
            raise RuntimeError(
                "Condensation returned a malformed video_script entry.")
        for field in ("narration", "visual"):
            if not isinstance(entry.get(field), str) \
                    or not entry[field].strip():
                raise RuntimeError(
                    "Condensation returned a video_script entry with "
                    f"empty {field!r}.")
    def str_list(value) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    return {
        "title": str(parsed.get("title", "") or "").strip(),
        "summary": str(parsed.get("summary", "") or "").strip(),
        "key_events": str_list(parsed.get("key_events")),
        "characters": str_list(parsed.get("characters")),
        "locations": str_list(parsed.get("locations")),
        "video_script": [{
            "narration": str(e.get("narration", "") or "").strip(),
            "visual": str(e.get("visual", "") or "").strip(),
            "source_reference": str(e.get("source_reference", "")
                                    or "").strip(),
        } for e in script if isinstance(e, dict)],
    }


def resolve_subject(name: str | None) -> dict | None:
    """Resolve --subject: preset dict, None for auto, error otherwise."""
    if name is None or str(name).strip().lower() == "auto":
        return None
    key = str(name).strip().lower()
    if key in SUBJECT_PRESETS:
        return dict(SUBJECT_PRESETS[key])
    raise ValueError(
        f"Unknown subject {name!r}; use 'auto' or one of: "
        f"{', '.join(sorted(SUBJECT_PRESETS))}")


def _normalize_subject(raw) -> dict:
    """Storyboard subject gate: complete profile or empty none-profile."""
    empty = {"type": "none", "identity": "", "features": "",
             "clothing_or_equipment": "", "consistency": ""}
    if not isinstance(raw, dict):
        return dict(empty)
    subject = {
        "type": str(raw.get("type", "none") or "none").strip(),
        "identity": str(raw.get("identity", "") or "").strip(),
        "features": str(raw.get("features", "") or "").strip(),
        "clothing_or_equipment": str(
            raw.get("clothing_or_equipment", "") or "").strip(),
        "consistency": str(raw.get("consistency", "") or "").strip(),
    }
    if subject["type"].lower() == "none" or not subject["identity"]:
        return dict(empty)
    return subject


def plan_storyboard(condensed: dict, *, model: str,
                    base_url: str = "http://127.0.0.1:11434",
                    num_scenes: int | None = None, timeout: int = 600,
                    subject: dict | None = None) -> dict:
    """PASS 2: condensed script + master subject -> storyboard dict."""
    from subject import assert_no_schema_leak

    if num_scenes is None:
        num_scenes = min(max(len(condensed.get("key_events", [])), 3), 10)
        print(f"Scene count derived from condensed script: {num_scenes}")
    master = dict(subject) if subject else None
    last_error: Exception | None = None
    for _ in range(1, MAX_ATTEMPTS + 1):
        # With a forced preset the model echoes it; on auto it proposes one.
        proposed = master or {"type": "auto", "identity": "",
                              "features": "", "clothing_or_equipment": "",
                              "consistency": ""}
        messages = [
            {"role": "system",
             "content": _storyboard_system(num_scenes, proposed)},
            {"role": "user", "content":
             "Condensed video script to convert into a storyboard:\n\n"
             + json.dumps(condensed, ensure_ascii=False)},
        ]
        try:
            parsed = _chat_json(messages, model=model, base_url=base_url,
                                timeout=timeout, what="storyboard planning")
        except RuntimeError as exc:
            last_error = exc
            continue
        try:
            scenes_raw = parsed.get("scenes")
            if not isinstance(scenes_raw, list) \
                    or len(scenes_raw) != num_scenes:
                raise ValueError(
                    f"expected exactly {num_scenes} scenes, got "
                    f"{len(scenes_raw) if isinstance(scenes_raw, list) else 'none'}")
            scenes = []
            for i, raw_scene in enumerate(scenes_raw, start=1):
                if not isinstance(raw_scene, dict):
                    raise ValueError(f"scene #{i} is not an object")
                image_prompt = raw_scene.get("image_prompt", "")
                video_prompt = raw_scene.get("video_prompt", "")
                narration = raw_scene.get("narration", "")
                if not isinstance(image_prompt, str) or not image_prompt.strip():
                    raise ValueError(f"scene #{i} has empty image_prompt")
                if not isinstance(video_prompt, str) or not video_prompt.strip():
                    raise ValueError(f"scene #{i} has empty video_prompt")
                if not isinstance(narration, str) or not narration.strip():
                    raise ValueError(f"scene #{i} has empty narration")
                assert_no_schema_leak(image_prompt)
                assert_no_schema_leak(video_prompt)
                try:
                    duration = int(raw_scene.get("duration")
                                   or DEFAULT_DURATION)
                except (TypeError, ValueError):
                    duration = DEFAULT_DURATION
                duration = min(max(duration, MIN_DURATION), MAX_DURATION)
                scenes.append({
                    "id": i,
                    "duration": duration,
                    "image_prompt": image_prompt.strip(),
                    "video_prompt": video_prompt.strip(),
                    "narration": narration.strip(),
                })
            final_subject = dict(master) if master \
                else _normalize_subject(parsed.get("subject"))
            return {"research": None, "subject": final_subject,
                    "scenes": scenes}
        except (ValueError, RuntimeError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Storyboard planning failed after {MAX_ATTEMPTS} attempts: "
        f"{last_error}")


def save_condensed_script(condensed: dict, path: str) -> None:
    """Persist the intermediate artifact (inspectable, resumable)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(condensed, f, indent=2, ensure_ascii=False)


def generate_from_script(path: str, *, model: str = "qwen3:8b",
                         base_url: str = "http://127.0.0.1:11434",
                         num_scenes: int | None = None, timeout: int = 600,
                         subject: str | None = "auto",
                         ) -> tuple[dict, dict, str]:
    """Source file -> (storyboard dict, condensed dict, source text).

    Text-only Qwen calls; never touches vision. Fails clearly instead of
    inventing scenes or events.
    """
    print(path);
    source_text = load_source_text(path)
    normalized = normalize_text(source_text)
    print(f"Loaded source: {len(normalized)} characters from {path}")
    ol.check_model_available(base_url, model)
    condensed = condense_source(normalized, model=model, base_url=base_url,
                                timeout=timeout)
    print(f"Condensed into {len(condensed['video_script'])} script beats, "
          f"{len(condensed['key_events'])} key events")
    preset = resolve_subject(subject)
    if preset is not None:
        print(f"Using configured subject profile "
              f"({str(subject).strip().lower()})")
    storyboard = plan_storyboard(condensed, model=model, base_url=base_url,
                                 num_scenes=num_scenes, timeout=timeout,
                                 subject=preset)
    print(f"Planned {len(storyboard['scenes'])} scenes from script")
    return storyboard, condensed, source_text
