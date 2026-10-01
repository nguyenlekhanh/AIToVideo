"""Generic subject profile: identity information shared by all scenes.

Model-agnostic and subject-agnostic (person, astronaut, animal, robot,
object...). Populated from the storyboard JSON; rendered into each scene's
image prompt by compose_scene_prompt(). Providers only ever see the final
prompt string plus the optional reference image.
"""
from __future__ import annotations

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
        """Render the identity block. Only non-empty parts are included."""
        parts = ["[RECURRING SUBJECT]"]
        if self.subject_type:
            parts.append(f"Type: {self.subject_type}.")
        if self.identity:
            parts.append(f"Identity: {self.identity}")
        if self.features:
            parts.append(f"Distinctive features: {self.features}")
        if self.clothing_or_equipment:
            parts.append(f"Clothing/equipment: {self.clothing_or_equipment}")
        if self.consistency:
            parts.append(f"Consistency: {self.consistency}")
        return "\n".join(parts)


REFERENCE_NOTE = (
    "[REFERENCE IMAGE] Match the identity and appearance of the provided "
    "reference image: preserve the same subject, features, clothing and "
    "colors. Change only the environment, action and composition described "
    "below; do not restyle or reinterpret the subject."
)


def compose_scene_prompt(subject: SubjectProfile | None, scene_description: str,
                         with_reference: bool = False) -> str:
    """Build the final image prompt: subject identity + scene + reference note.

    With no subject (old storyboards), the scene description is returned
    unchanged to preserve existing behavior.
    """
    scene = (scene_description or "").strip()
    if subject is None or subject.is_empty():
        return scene
    blocks = [subject.render(), f"[SCENE]\n{scene}"]
    if with_reference:
        blocks.append(REFERENCE_NOTE)
    return "\n\n".join(blocks)
