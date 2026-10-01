"""Unit tests: state manifest + output validation (no GPU, no network)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import state as st

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
JPEG = (b"\xff\xd8\xff\xe0" + b"\x00" * 100)


class ManifestTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            state = st.new_state("demo")
            state["scene_ids"] = [1, 2, 3]
            state["base_seed"] = 42
            st.mark_scene_complete(state, "image", 2)
            st.mark_scene_complete(state, "image", 2)
            st.mark_stage_complete(state, "storyboard")
            st.save_state(path, state)
            loaded = st.load_state(path, "demo")
            self.assertEqual(loaded["project"], "demo")
            self.assertEqual(loaded["base_seed"], 42)
            self.assertEqual(loaded["scene_ids"], [1, 2, 3])
            self.assertEqual(st.completed_scenes(loaded, "image"), [2])
            self.assertEqual(st.scene_attempts(loaded, "image", 2), 2)
            self.assertEqual(st.stage_status(loaded, "storyboard"), "complete")
            self.assertEqual(st.stage_status(loaded, "image"), "partial")
            self.assertEqual(st.stage_status(loaded, "video"), "pending")

    def test_missing_file_fresh_skeleton(self):
        state = st.load_state(os.path.join("nope", "state.json"), "p")
        self.assertEqual(state["project"], "p")
        self.assertEqual(st.stage_status(state, "image"), "pending")

    def test_corrupt_file_fresh_skeleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as f:
                f.write("{not json")
            state = st.load_state(path, "p")
            self.assertEqual(st.stage_status(state, "image"), "pending")

    def test_stage_complete_partial_pending(self):
        state = st.new_state("p")
        state["scene_ids"] = [1, 2]
        self.assertEqual(st.stage_status(state, "image"), "pending")
        st.mark_scene_complete(state, "image", 1)
        self.assertEqual(st.stage_status(state, "image"), "partial")
        st.mark_scene_complete(state, "image", 2)
        self.assertEqual(st.stage_status(state, "image"), "complete")


class ValidationTest(unittest.TestCase):
    def _file(self, tmp, name, content):
        path = os.path.join(tmp, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_valid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(st.valid_image_file(self._file(tmp, "a.png", PNG)))
            self.assertTrue(st.valid_image_file(self._file(tmp, "b.jpg", JPEG)))
            self.assertFalse(st.valid_image_file(self._file(tmp, "c.png", b"nope")))
            self.assertFalse(st.valid_image_file(self._file(tmp, "d.png", b"")))
            self.assertFalse(st.valid_image_file(os.path.join(tmp, "missing.png")))

    def test_valid_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(st.valid_audio_file(self._file(tmp, "a.mp3", b"ID3" + b"\x00" * 50)))
            self.assertFalse(st.valid_audio_file(self._file(tmp, "b.mp3", b"")))
            self.assertFalse(st.valid_audio_file(os.path.join(tmp, "missing.mp3")))

    def test_video_validation_with_proof_assets(self):
        clip = os.path.join(AI_VIDEO_DIR, "projects", "mars_city",
                            "videos", "scene_001.mp4")
        if not os.path.isfile(clip):
            self.skipTest("proof assets not present")
        self.assertTrue(st.valid_video_file(clip))
        self.assertTrue(st.valid_video_file(clip, 5.0))
        self.assertFalse(st.valid_video_file(clip, 30.0))
        self.assertFalse(st.valid_video_file(os.path.join("nope.mp4"), 5.0))

    def test_probe_missing_returns_none(self):
        self.assertIsNone(st.probe_video_duration(os.path.join("nope.mp4")))


if __name__ == "__main__":
    unittest.main()
