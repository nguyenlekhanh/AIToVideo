"""Storyboard validation and persistence. JSON is validated before continuing."""
from __future__ import annotations

import json
import os

from subject import SubjectProfile

REQUIRED_FIELDS = ("id", "duration", "image_prompt", "video_prompt", "narration")


def validate_storyboard(data: dict) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Storyboard must be a JSON object with a 'scenes' list.")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard must contain a non-empty 'scenes' list.")
    validated: list[dict] = []
    for i, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene #{i} must be an object.")
        for field in REQUIRED_FIELDS:
            if field == "id":
                continue  # ids are normalized to 1..N below; accept if missing
            if field not in scene:
                raise ValueError(f"Scene #{i} is missing required field: {field!r}")
        try:
            scene_id = int(scene.get("id", i))
        except (TypeError, ValueError):
            scene_id = i
        try:
            duration = int(scene["duration"])
        except (TypeError, ValueError):
            raise ValueError(f"Scene #{i} has non-integer duration: {scene['duration']!r}")
        if duration <= 0:
            raise ValueError(f"Scene #{i} has non-positive duration: {duration}")
        for field in ("image_prompt", "video_prompt", "narration"):
            if not isinstance(scene[field], str) or not scene[field].strip():
                raise ValueError(f"Scene #{i} field {field!r} must be a non-empty string.")
        validated.append(
            {
                "id": scene_id,
                "duration": duration,
                "image_prompt": scene["image_prompt"].strip(),
                "video_prompt": scene["video_prompt"].strip(),
                "narration": scene["narration"].strip(),
            }
        )
    # normalize ids to 1..N in order
    for i, scene in enumerate(validated, start=1):
        scene["id"] = i
    return validated


def save_storyboard(scenes: list[dict], path: str, subject=None) -> None:
    """Save scenes plus the optional SubjectProfile (SubjectProfile or dict).

    Old files without "subject" still load via load_storyboard/load_subject.
    """
    if subject is not None and not isinstance(subject, dict):
        to_dict = getattr(subject, "to_dict", None)
        subject = to_dict() if callable(to_dict) else dict(subject)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"subject": subject, "scenes": scenes}, f, indent=2,
                  ensure_ascii=False)


def load_storyboard(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return validate_storyboard(json.load(f))


def load_subject(path: str) -> SubjectProfile | None:
    """Restore the SubjectProfile from a saved storyboard (None if absent)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return SubjectProfile.from_dict(data.get("subject"))
