"""Ollama client: generate a structured JSON storyboard. No manual prose parsing."""
from __future__ import annotations

import json
import re
import urllib.request

SYSTEM_PROMPT = """You are a storyboard generator for a short AI video.
Output ONLY valid JSON. No markdown, no code fences, no commentary.
Schema:
{"subject": {"type": "...", "identity": "...", "features": "...", "clothing_or_equipment": "...", "consistency": "..."}, "scenes": [{"id": 1, "duration": 5, "image_prompt": "...", "video_prompt": "...", "narration": "..."}]}
Rules:
- id starts at 1 and increments by 1.
- duration is an integer number of seconds, 3 to 8.
- image_prompt: detailed text-to-image prompt, one keyframe, visual style, no narration text.
- video_prompt: short image-to-video motion description (camera move, subject motion).
- narration: one or two sentences of voiceover, plain text, no stage directions.
- subject: the single recurring visual subject (a person, astronaut, child, animal, robot, object, ...) described in generic identity terms so it stays recognizable across scenes: physical appearance, distinctive features/marks, clothing/equipment/colors. Omit fields that do not apply; use {} if there is no single recurring subject.
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


def generate_storyboard(
    user_prompt: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    num_scenes: int = 3,
    timeout: int = 180,
) -> dict:
    """Ask Ollama for a storyboard. Returns parsed dict, not prose."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Create a storyboard with exactly {num_scenes} scenes "
                    f"for this video idea: {user_prompt}\n"
                    "Return ONLY the JSON object."
                ),
            },
        ],
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
