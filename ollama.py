"""Ollama client: generate a structured JSON storyboard. No manual prose parsing."""
from __future__ import annotations

import json
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
