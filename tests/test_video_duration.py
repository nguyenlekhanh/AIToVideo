"""Regression tests: storyboard.scene.duration must reach the video request.

No GPU/video generation: mocked providers and stubbed ComfyUI transport.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import main
from providers.comfy import ComfyClient
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = 5.0


def scenes_567():
    return [{"id": 1, "duration": 5}, {"id": 2, "duration": 6},
            {"id": 3, "duration": 7}]


class ResolveDurationTest(unittest.TestCase):
    def test_scene_values_preserved(self):
        self.assertEqual(
            [main.resolve_scene_duration(s, DEFAULT) for s in scenes_567()],
            [5.0, 6.0, 7.0])

    def test_seven_is_not_five(self):
        self.assertNotEqual(
            main.resolve_scene_duration({"duration": 7}, DEFAULT), 5.0)

    def test_missing_duration_uses_default(self):
        self.assertEqual(main.resolve_scene_duration({}, DEFAULT), DEFAULT)
        self.assertEqual(main.resolve_scene_duration({"duration": None}, DEFAULT),
                         DEFAULT)

    def test_invalid_duration_uses_default(self):
        for bad in (0, -3, "long", [5], {"d": 5}):
            with self.subTest(bad=bad):
                self.assertEqual(
                    main.resolve_scene_duration({"duration": bad}, DEFAULT),
                    DEFAULT)

    def test_float_duration_accepted(self):
        self.assertEqual(main.resolve_scene_duration({"duration": 6.5}, DEFAULT),
                         6.5)


class StubClient(ComfyClient):
    """Captures queued workflows; writes no files, touches no server."""

    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.queued = []

    def upload_image(self, image_path: str) -> str:
        return os.path.basename(image_path)

    def run(self, workflow, output_keys, dest_path: str) -> str:
        self.queued.append(workflow)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"fake")
        return dest_path


class LtxDurationRequestTest(unittest.TestCase):
    def _provider(self):
        return create_provider("video", AI_VIDEO_DIR, "ltx",
                               client=StubClient())

    def _request(self, duration, seed=1):
        return VideoRequest(prompt="p", negative_prompt="n",
                            input_image=Path(__file__), width=768, height=512,
                            duration=duration, seed=seed,
                            output_path=Path("out.mp4"))

    def test_requested_durations_reach_workflow(self):
        import tempfile
        provider = self._provider()
        for seconds in (5, 6, 7):
            with self.subTest(seconds=seconds):
                with tempfile.TemporaryDirectory() as tmp:
                    req = self._request(seconds)
                    req = VideoRequest(
                        prompt=req.prompt, negative_prompt=req.negative_prompt,
                        input_image=req.input_image, width=req.width,
                        height=req.height, duration=req.duration, seed=req.seed,
                        output_path=Path(tmp) / "out.mp4")
                    provider.client.queued.clear()
                    provider.generate(req)
                    wf = provider.client.queued[-1]
                    found = provider._find_nodes(wf)
                    self.assertEqual(wf[found["duration"]]["inputs"]["value"],
                                     seconds)

    def test_out_of_range_rejected(self):
        provider = self._provider()
        with self.assertRaises(ValueError):
            provider.generate(self._request(99))

    def test_three_scene_integration(self):
        """Phase 7: 5/6/7 storyboard scenes -> 5/6/7 provider requests."""
        import tempfile
        provider = self._provider()
        with tempfile.TemporaryDirectory() as tmp:
            requested = []
            for i, scene in enumerate(scenes_567(), start=1):
                seconds = main.resolve_scene_duration(scene, DEFAULT)
                requested.append(seconds)
                provider.generate(VideoRequest(
                    prompt="p", negative_prompt="n",
                    input_image=Path(__file__), width=768, height=512,
                    duration=seconds, seed=i,
                    output_path=Path(tmp) / f"scene_{i:03d}.mp4"))
            self.assertEqual(requested, [5.0, 6.0, 7.0])
            for wf, seconds in zip(provider.client.queued, (5, 6, 7)):
                found = provider._find_nodes(wf)
                self.assertEqual(wf[found["duration"]]["inputs"]["value"], seconds)


class WanDurationConstraintTest(unittest.TestCase):
    def test_frames_math_documents_approximation(self):
        from providers.video.wan import WanVideoProvider
        # Wan lattice is 4k+1 frames @16fps: requested durations map to the
        # nearest lattice point, reported honestly (never claimed exact).
        self.assertEqual(WanVideoProvider.frames_for_duration(5.0), 81)    # 5.06s
        self.assertEqual(WanVideoProvider.frames_for_duration(6.0), 97)    # 6.06s
        self.assertEqual(WanVideoProvider.frames_for_duration(7.0), 113)   # 7.06s
        for seconds, frames in ((5.0, 81), (6.0, 97), (7.0, 113)):
            self.assertEqual((frames - 1) % 4, 0)
            actual = frames / 16.0
            self.assertLess(abs(actual - seconds), 0.1)


if __name__ == "__main__":
    unittest.main()
