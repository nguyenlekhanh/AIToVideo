"""Unit tests: storyboard validation (missing id tolerated, content required)."""
from __future__ import annotations

import json
import os
import unittest

import storyboard as sb


def scene(**overrides):
    base = {"id": 1, "duration": 5, "image_prompt": "a cat",
            "video_prompt": "pan", "narration": "A cat."}
    base.update(overrides)
    return base


class StoryboardTest(unittest.TestCase):
    def test_valid(self):
        scenes = sb.validate_storyboard({"scenes": [scene(), scene(id=2)]})
        self.assertEqual([s["id"] for s in scenes], [1, 2])

    def test_missing_id_tolerated(self):
        raw = {"scenes": [scene(), {k: v for k, v in scene().items() if k != "id"}]}
        scenes = sb.validate_storyboard(raw)
        self.assertEqual([s["id"] for s in scenes], [1, 2])

    def test_non_integer_id_tolerated(self):
        raw = {"scenes": [scene(id="first")]}
        self.assertEqual(sb.validate_storyboard(raw)[0]["id"], 1)

    def test_ids_normalized(self):
        raw = {"scenes": [scene(id=9), scene(id=3)]}
        self.assertEqual([s["id"] for s in sb.validate_storyboard(raw)], [1, 2])

    def test_missing_narration_rejected(self):
        raw = {"scenes": [{k: v for k, v in scene().items() if k != "narration"}]}
        with self.assertRaises(ValueError):
            sb.validate_storyboard(raw)

    def test_empty_scenes_rejected(self):
        with self.assertRaises(ValueError):
            sb.validate_storyboard({"scenes": []})

    def test_bad_duration_rejected(self):
        with self.assertRaises(ValueError):
            sb.validate_storyboard({"scenes": [scene(duration=0)]})
        with self.assertRaises(ValueError):
            sb.validate_storyboard({"scenes": [scene(duration="long")]})

    def test_save_and_reload_subject(self):
        import tempfile

        from subject import SubjectProfile
        profile = SubjectProfile.from_dict({
            "type": "astronaut",
            "identity": "an adult astronaut in a white EVA suit",
            "features": "gold visor",
            "clothing_or_equipment": "white spacesuit",
            "consistency": "preserve suit design",
        })
        assert profile is not None
        scenes = sb.validate_storyboard({"scenes": [scene(), scene(id=2)]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "storyboard.json")
            sb.save_storyboard(scenes, path, profile)
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["subject"]["type"], "astronaut")
            self.assertEqual(len(raw["scenes"]), 2)
            restored = sb.load_subject(path)
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.to_dict(), profile.to_dict())
            self.assertEqual(sb.load_storyboard(path), scenes)

    def test_save_without_subject(self):
        import tempfile

        scenes = sb.validate_storyboard({"scenes": [scene()]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "storyboard.json")
            sb.save_storyboard(scenes, path)
            self.assertIsNone(sb.load_subject(path))
            self.assertEqual(sb.load_storyboard(path), scenes)

    def test_old_storyboard_without_subject(self):
        import tempfile

        scenes = sb.validate_storyboard({"scenes": [scene()]})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"scenes": scenes}, f)
            self.assertIsNone(sb.load_subject(path))
            self.assertEqual(sb.load_storyboard(path), scenes)


if __name__ == "__main__":
    unittest.main()
