"""Ollama client: generate a structured JSON storyboard. No manual prose parsing."""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.request

SYSTEM_PROMPT = """You are a storyboard generator for a short AI video.
Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"subject": {"type": "...", "identity": "...", "features": "...", "clothing_or_equipment": "...", "consistency": "..."}, "scenes": [{"id": 1, "duration": 5, "image_prompt": "...", "video_prompt": "...", "narration": "...", "research_fact_ids": ["fact_001"], "source_ids": ["src_001"], "grounding": "researched"}]}
Rules:
- id starts at 1 and increments by 1.
- duration is an integer number of seconds, 3 to 8.
- image_prompt: detailed text-to-image prompt, one keyframe, visual style, no narration text.
- video_prompt: short image-to-video motion description (camera move, subject motion).
- narration: one or two sentences of voiceover, plain text, no stage directions.
- subject: the single recurring visual subject (a person, astronaut, child, animal, robot, object, ...) described in generic identity terms so it stays recognizable across scenes: physical appearance, distinctive features/marks, clothing/equipment/colors. Omit fields that do not apply; use {} if there is no single recurring subject.
- When a "Web research context" message is provided, it contains numbered facts ([fact_001]...), disagreements ([conflict_001]...) and sources ([src_001]...): use researched facts for factual claims, do not invent facts, citations, URLs or dates, and reference facts/sources ONLY by their given IDs. Every scene whose narration makes factual claims must include research_fact_ids (and source_ids); phrase narrations as close paraphrases of the cited facts so each claim stays traceable; prefer recent facts when the topic asks for latest/current information; do not use irrelevant research just because it exists; never turn search noise into facts. Scenes with no factual claims use "grounding": "creative". Creative visual details are allowed only when they do not contradict the research. If facts disagree (a conflict entry), do not present the disputed detail as unquestioned fact.
- Exactly the requested number of scenes."""


def _post(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_reachable(base_url: str, timeout: int = 10) -> None:
    """Raise RuntimeError if Ollama is not reachable."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout) as resp:
            resp.read()
    except Exception as exc:
        raise RuntimeError(f"Ollama not reachable at {base_url}: {exc}") from exc


def _extract_json(text: str) -> dict:
    """Parse model output as JSON, tolerating fences / leading prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # strip markdown fences
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Ollama did not return JSON: {text[:500]!r}")
    return json.loads(m.group(0))


def rewrite_scene_narration(scene: dict, facts: list[dict], reason: str,
                            model: str, base_url: str = "http://127.0.0.1:11434",
                            timeout: int = 180) -> dict:
    """Rewrite ONLY a scene's narration so cited facts support it.

    facts: [{id, claim, ...}] already cited by the scene (never invent new
    fact/source IDs). Returns a copy of the scene with a new narration;
    every other field is preserved byte-for-byte. Raises RuntimeError on
    failure so the caller fails clearly instead of silently passing.
    """
    claims = "\n".join(f"- [{fact.get('id')}] {fact.get('claim', '')}"
                       for fact in facts)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content":
             "You rewrite one storyboard scene narration. Output ONLY valid "
             "JSON: {\"narration\": \"...\"}. No markdown, no commentary."},
            {"role": "user", "content":
             f"Scene #{scene.get('id')} current narration:\n"
             f"{scene.get('narration', '')}\n\n"
             f"Cited research facts (use ONLY these, close paraphrase):\n"
             f"{claims}\n\n"
             f"Problem with the current narration: {reason}\n\n"
             "Rewrite the narration in one or two sentences so every factual "
             "claim is directly supported by the cited facts above. Do not "
             "introduce new facts, numbers, dates, names, or superlatives. "
             "Keep the same research_fact_ids and source_ids (do not invent "
             "new IDs). Return ONLY the JSON object."},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.5},
    }
    try:
        result = _post(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
    except Exception as exc:
        raise RuntimeError(f"Ollama rewrite failed (model={model}): {exc}") from exc
    content = (result.get("message") or {}).get("content", "")
    try:
        parsed = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama rewrite returned invalid JSON: {exc}\nRaw: {content[:500]}"
        ) from exc
    narration = (parsed.get("narration") or "").strip() if isinstance(parsed, dict) else ""
    if not narration:
        raise RuntimeError(
            f"Ollama rewrite returned empty narration. Raw: {content[:500]}")
    updated = dict(scene)
    updated["narration"] = narration
    return updated


def generate_storyboard(
    user_prompt: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    num_scenes: int = 3,
    timeout: int = 180,
    research_context: str | None = None,
) -> dict:
    """Ask Ollama for a storyboard. Returns parsed dict, not prose.

    research_context, when given, is appended as a SEPARATE labeled message
    after the user's original prompt (which is never modified or mixed).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Create a storyboard with exactly {num_scenes} scenes "
                f"for this video idea: {user_prompt}\n"
                "Return ONLY the JSON object."
            ),
        },
    ]
    if research_context and research_context.strip():
        messages.append({
            "role": "user",
            "content": (
                "Web research context for the topic above. "
                "Base the storyboard on these researched facts where relevant; "
                "do not invent newer facts beyond what is stated here.\n\n"
                + research_context.strip()
            ),
        })
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.7},
    }
    try:
        result = _post(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed (model={model}): {exc}") from exc
    content = (result.get("message") or {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned empty response: {str(result)[:500]}")
    try:
        return _extract_json(content)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {exc}\nRaw: {content[:1000]}") from exc


# ---------------------------------------------------------------------------
# Mode B: keyframe-to-storyboard visual analysis (Qwen-VL via Ollama).
#
# One keyframe image = one storyboard scene. Every image is sent as base64
# in the Ollama chat `images` field so Qwen-VL actually inspects it; the
# filename alone is never the visual source. Generic prompts only: no
# per-topic wording anywhere (landscapes, cats, people, products, ... all
# flow through the same instruction).
# ---------------------------------------------------------------------------

KEYFRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
KEYFRAME_SCENE_DURATION = 7
KEYFRAME_MAX_ATTEMPTS = 3

KEYFRAME_SYSTEM_PROMPT = """You analyze ONE keyframe image and convert it into exactly ONE storyboard scene.
Ground every word strictly in what is visibly present in the image. Do not invent subjects, objects, locations, events, or themes that cannot reasonably be inferred from the image. Do not replace the visible subject with a more imaginative one. Do not treat the image as mere creative inspiration. If something is unclear, describe it conservatively.

Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"image_prompt": "...", "video_prompt": "...", "subject": {"type": "...", "identity": "...", "features": "...", "clothing_or_equipment": "...", "consistency": "..."}}

Rules:
- image_prompt: pure visual prose describing the frame to recreate (main subject, environment, composition, camera angle and distance, lighting, weather/time of day, colors, important foreground/background objects, visual style). No schema labels, no JSON keys, no scene IDs, no instructions.
- video_prompt: plausible motion based ONLY on elements visible in the image (camera movement plus natural movement of water, vegetation, clouds, people, animals, vehicles, etc.). Do not invent actions that contradict the image.
- subject: describe a recurring subject ONLY when the image clearly shows one that must stay consistent (a specific person, animal, product, ...). For visual-only content (landscapes, waterfalls, rivers, flowers, ocean, architecture, food, cars, animals, objects, scenery) use {"type": "none", "identity": "", "features": "", "clothing_or_equipment": "", "consistency": ""}. Never force a human subject that is not visible."""


def _natural_sort_key(path: str) -> list:
    """Split on digit runs so 2.jpg sorts before 10.jpg."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", os.path.basename(path))]


def discover_keyframes(directory: str) -> list[str]:
    """Find supported keyframe images, deterministically natural-sorted.

    Case-insensitive extensions; unsupported files ignored. Raises
    FileNotFoundError when the directory is missing, ValueError with the
    exact message 'No supported keyframe images found in <directory>'
    when nothing usable is present.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Keyframe directory not found: {directory}")
    found = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in KEYFRAME_EXTENSIONS:
            found.append(path)
    if not found:
        raise ValueError(f"No supported keyframe images found in {directory}")
    found.sort(key=_natural_sort_key)
    return found


def load_keyframe_b64(path: str) -> str:
    """Read image bytes and return base64. Fails loudly, never empty."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        raise ValueError(f"Cannot read keyframe image {path}: {exc}") from exc
    if not raw:
        raise ValueError(f"Keyframe image is empty: {path}")
    return base64.b64encode(raw).decode("ascii")


def check_model_available(base_url: str, model: str, timeout: int = 10) -> None:
    """Fail clearly when Ollama or the analysis model is missing.

    Never downloads anything; tells the user the exact pull command.
    """
    try:
        with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/api/tags",
                timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Ollama not reachable at {base_url}: {exc}") from exc
    names = set()
    for entry in data.get("models", []) or []:
        name = entry.get("name", "")
        names.add(name)
        names.add(name.split(":")[0])
    if model not in names and model.split(":")[0] not in names:
        raise RuntimeError(
            f"Analysis model {model!r} is not available on Ollama at "
            f"{base_url}. Pull it with: ollama pull {model}")


def analyze_keyframe(image_b64: str, *, model: str,
                     base_url: str = "http://127.0.0.1:11434",
                     timeout: int = 600, label: str = "keyframe",
                     max_attempts: int = KEYFRAME_MAX_ATTEMPTS) -> dict:
    """Send ONE image to Qwen-VL; return its scene JSON (bounded retries).

    The image travels in the chat `images` field (base64). Malformed model
    JSON and schema-leaking prompts are retried up to max_attempts times,
    then a clear error naming the keyframe is raised. Never fabricates.
    """
    from subject import assert_no_schema_leak

    user_content = (
        "Analyze the provided keyframe image and create exactly ONE "
        "storyboard scene grounded strictly in what is visibly present "
        "in the image. Return ONLY the JSON object.")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": KEYFRAME_SYSTEM_PROMPT},
                {"role": "user", "content": user_content,
                 "images": [image_b64]},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            result = _post(f"{base_url.rstrip('/')}/api/chat", payload, timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Qwen-VL analysis failed for {label} (model={model}): "
                f"{exc}") from exc
        content = (result.get("message") or {}).get("content", "")
        if not content:
            last_error = ValueError("empty response")
            continue
        try:
            parsed = _extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            last_error = ValueError("response is not a JSON object")
            continue
        image_prompt = parsed.get("image_prompt", "")
        video_prompt = parsed.get("video_prompt", "")
        if (not isinstance(image_prompt, str) or not image_prompt.strip()
                or not isinstance(video_prompt, str) or not video_prompt.strip()):
            last_error = ValueError("missing image_prompt/video_prompt")
            continue
        try:
            assert_no_schema_leak(image_prompt)
            assert_no_schema_leak(video_prompt)
        except ValueError as exc:
            last_error = exc
            continue
        subject = parsed.get("subject") or {}
        if not isinstance(subject, dict):
            subject = {}
        return {
            "image_prompt": image_prompt.strip(),
            "video_prompt": video_prompt.strip(),
            "subject": {
                "type": str(subject.get("type", "none") or "none").strip(),
                "identity": str(subject.get("identity", "") or "").strip(),
                "features": str(subject.get("features", "") or "").strip(),
                "clothing_or_equipment": str(
                    subject.get("clothing_or_equipment", "") or "").strip(),
                "consistency": str(subject.get("consistency", "") or "").strip(),
            },
        }
    raise RuntimeError(
        f"Qwen-VL analysis failed for {label} after {max_attempts} "
        f"attempts (malformed output): {last_error}")


def generate_storyboard_from_keyframes(
    directory: str, *, model: str = "qwen3-vl:8b",
    base_url: str = "http://127.0.0.1:11434",
    num_scenes: int | None = None, timeout: int = 600,
) -> dict:
    """Keyframe directory -> storyboard dict in the existing schema.

    One keyframe = one scene. num_scenes (explicit --scenes) smaller than
    the keyframe count uses the first N; larger fails clearly instead of
    inventing scenes; None uses every keyframe.
    """
    paths = discover_keyframes(directory)
    print(f"Found {len(paths)} keyframes")
    if num_scenes is not None:
        if num_scenes > len(paths):
            raise ValueError(
                f"Requested {num_scenes} scenes but only {len(paths)} "
                f"keyframes found in {directory}; refusing to invent "
                f"additional scenes.")
        if num_scenes < len(paths):
            print(f"Using first {num_scenes} of {len(paths)} keyframes "
                  f"(--scenes {num_scenes})")
            paths = paths[:num_scenes]
    print(f"Using {len(paths)} scenes from keyframes")
    check_model_available(base_url, model)
    analyses = []
    for i, path in enumerate(paths, start=1):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        print(f"[{i}/{len(paths)}] Keyframe: {os.path.basename(path)} "
              f"({size} bytes, model={model}, image attached)")
        image_b64 = load_keyframe_b64(path)
        print(f"  Analyzing keyframe...")
        analyses.append(analyze_keyframe(
            image_b64, model=model, base_url=base_url, timeout=timeout,
            label=f"keyframe {os.path.basename(path)}"))
    scenes = []
    for i, analysis in enumerate(analyses, start=1):
        scenes.append({
            "id": i,
            "duration": KEYFRAME_SCENE_DURATION,
            "image_prompt": analysis["image_prompt"],
            "video_prompt": analysis["video_prompt"],
            "narration": "",
        })
    # Global subject only on unanimous non-none agreement; else none.
    # This prevents injecting a recurring person into visual-only sequences.
    subject = {"type": "none", "identity": "", "features": "",
               "clothing_or_equipment": "", "consistency": ""}
    if analyses:
        types = {a["subject"]["type"].strip().lower() for a in analyses}
        identities = {a["subject"]["identity"].strip().lower()
                      for a in analyses}
        if len(types) == 1 and len(identities) == 1:
            only_type = next(iter(types))
            only_identity = next(iter(identities))
            if only_type and only_type != "none":
                subject = dict(analyses[0]["subject"])
    return {"research": None, "subject": subject, "scenes": scenes}
