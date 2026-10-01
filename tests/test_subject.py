"""Unit tests: generic SubjectProfile (no hard-coded characters)."""
from __future__ import annotations

import unittest

import ollama as ol
from subject import SubjectProfile, compose_scene_prompt


ASTRONAUT = {
    "type": "astronaut",
    "identity": "an adult astronaut in a white EVA suit",
    "features": "reflective gold visor, rectangular mission patch on the chest",
    "clothing_or_equipment": "white spacesuit, backpack life-support unit",
    "consistency": "preserve the same suit design, proportions and markings",
}


class SubjectProfileTest(unittest.TestCase):
    def test_from_dict_full(self):
        profile = SubjectProfile.from_dict(dict(ASTRONAUT))
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.subject_type, "astronaut")
        self.assertIn("gold visor", profile.features)

    def test_from_dict_partial_animal(self):
        profile = SubjectProfile.from_dict({
            "type": "animal",
            "identity": "a red fox",
            "features": "white-tipped tail, black ear backs",
        })
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile.clothing_or_equipment, "")

    def test_from_dict_absent(self):
        self.assertIsNone(SubjectProfile.from_dict(None))
        self.assertIsNone(SubjectProfile.from_dict({}))
        self.assertIsNone(SubjectProfile.from_dict({"type": "  "}))
        self.assertIsNone(SubjectProfile.from_dict("not-a-dict"))

    def test_render_skips_empty(self):
        profile = SubjectProfile.from_dict({
            "type": "robot", "identity": "a boxy service robot"})
        assert profile is not None
        text = profile.render()
        self.assertIn("robot", text)
        self.assertIn("boxy service robot", text)
        self.assertNotIn("Distinctive features", text)
        self.assertNotIn("Clothing", text)

    def test_compose_no_subject_unchanged(self):
        scene = "A robot walks through a market."
        self.assertEqual(compose_scene_prompt(None, scene), scene)
        self.assertEqual(
            compose_scene_prompt(SubjectProfile.from_dict({}), scene), scene)

    def test_compose_includes_identity_scene_and_note(self):
        profile = SubjectProfile.from_dict(dict(ASTRONAUT))
        text = compose_scene_prompt(profile, "The astronaut climbs a dune.",
                                    with_reference=True)
        self.assertIn("gold visor", text)
        self.assertIn("The astronaut climbs a dune.", text)
        self.assertIn("reference image", text.lower())

    def test_compose_no_reference_note(self):
        profile = SubjectProfile.from_dict(dict(ASTRONAUT))
        text = compose_scene_prompt(profile, "The astronaut climbs a dune.",
                                    with_reference=False)
        self.assertIn("gold visor", text)
        self.assertNotIn("REFERENCE IMAGE", text)

    def test_storyboard_raw_without_subject(self):
        self.assertIsNone(SubjectProfile.from_dict({}.get("subject")))

    def test_ollama_schema_mentions_subject(self):
        self.assertIn("subject", ol.SYSTEM_PROMPT)
        self.assertIn("scenes", ol.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
