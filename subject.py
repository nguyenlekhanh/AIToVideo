"""Generic subject profile: identity information shared by all scenes.

Model-agnostic and subject-agnostic (person, astronaut, animal, robot,
object...). Populated from the storyboard JSON; rendered into each scene's
image prompt by compose_scene_prompt(). Providers only ever see the final
prompt string plus the optional reference image.

CRITICAL: the rendered prompt is natural-language visual description ONLY.
Schema labels, keys, brackets, IDs and instructions must never reach an
image model (they get rendered as visible text). assert_no_schema_leak()
enforces this on every composed prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SubjectProfile:
    subject_type: str = ""
    identity: str = ""
    features: str = ""
    clothing_or_equipment: str = ""
    consistency: str = ""

    @classmethod
    def from_dict(cls, data: dict | None) -> "SubjectProfile | None":
        """Build a profile from storyboard JSON. Returns None when absent."""
        if not isinstance(data, dict):
            return None
        profile = cls(
            subject_type=str(data.get("type", "") or "").strip(),
            identity=str(data.get("identity", "") or "").strip(),
            features=str(data.get("features", "") or "").strip(),
            clothing_or_equipment=str(data.get("clothing_or_equipment", "") or "").strip(),
            consistency=str(data.get("consistency", "") or "").strip(),
        )
        return None if profile.is_empty() else profile

    def is_empty(self) -> bool:
        return not (self.subject_type or self.identity or self.features
                    or self.clothing_or_equipment or self.consistency)

    def to_dict(self) -> dict:
        """Serialize for storyboard.json (inverse of from_dict)."""
        return {
            "type": self.subject_type,
            "identity": self.identity,
            "features": self.features,
            "clothing_or_equipment": self.clothing_or_equipment,
            "consistency": self.consistency,
        }

    def render(self) -> str:
        """Render a natural-language visual description of the subject.

        Comma-joined visual phrases, no labels/keys/brackets/instructions.
        Only identity/features/clothing contribute (type is redundant when
        identity exists; consistency notes are non-visual meta-instructions
        and are intentionally excluded).
        """
        base = self.identity.strip()
        if not base and self.subject_type.strip():
            base = f"a {self.subject_type.strip()}"
        if not base:
            return ""
        extras = []
        if self.features.strip():
            extras.append(self.features.strip().rstrip("."))
        if self.clothing_or_equipment.strip():
            extras.append(
                f"wearing {self.clothing_or_equipment.strip()}".rstrip("."))
        description = base if not extras else f"{base}, {', '.join(extras)}"
        description = description[0].upper() + description[1:] + "."
        return description


# Internal-schema leakage signatures. Snake_case keys and JSON-ish field
# names can never occur in natural prose, so they match anywhere; plain
# prose words (subject/scene/...) only match as line-leading labels.
SCHEMA_LEAK_PATTERNS = (
    r"^\s*[\[\(]?(?:recurring subject|subject|scene|reference image)[\]\)]?\s*:",
    r"(?<!\w)(?:image_prompt|video_prompt|narration|duration|"
    r"research_fact_ids|source_ids|grounding|clothing_or_equipment|"
    r"consistency|identity|features|type)[\"']?\s*:",
)
_SCHEMA_LEAK = re.compile("|".join(SCHEMA_LEAK_PATTERNS),
                          re.IGNORECASE | re.MULTILINE)


def assert_no_schema_leak(prompt: str) -> str:
    """Fail loudly if an image prompt contains storyboard schema content.

    Targets serialized/schema formatting, not normal English: "scene" or
    "features" inside ordinary sentences pass; "Scene:" labels and
    "clothing_or_equipment:" keys do not.
    """
    matches = sorted(set(_SCHEMA_LEAK.findall(prompt or "")))
    if matches:
        raise ValueError(
            "Image prompt contains internal storyboard schema content "
            f"({', '.join(matches)}); refusing to send it to the image "
            "provider. Prompt must be natural-language visual description.")
    return prompt


def compose_scene_prompt(subject: SubjectProfile | None, scene_description: str,
                         with_reference: bool = False) -> str:
    """Build the final image prompt: natural visual description only.

    Subject identity is woven in as flowing visual phrases
    ("A young woman with long black hair, wearing a white ao dai."),
    followed by the scene description. No labels, keys, brackets, IDs or
    instructions are ever emitted. with_reference is accepted for API
    compatibility; reference identity travels via the reference IMAGE
    (img2img latent), never via words. Guarded by assert_no_schema_leak().

    With no subject (old storyboards), the scene description is returned
    unchanged to preserve existing behavior.
    """
    scene = (scene_description or "").strip()
    if subject is None or subject.is_empty():
        return assert_no_schema_leak(scene)
    description = subject.render()
    composed = f"{description} {scene}" if description else scene
    return assert_no_schema_leak(composed)
