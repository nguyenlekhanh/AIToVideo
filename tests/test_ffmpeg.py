"""Unit tests: ffmpeg mux uses finite loops and trims to narration length."""
from __future__ import annotations

import os
import unittest

import ffmpeg as ff

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARS = os.path.join(AI_VIDEO_DIR, "projects", "mars_city")


class MuxTest(unittest.TestCase):
    def test_mux_trims_to_audio(self):
        clip = os.path.join(MARS, "videos", "scene_001.mp4")
        audio = os.path.join(MARS, "audio", "scene_001.mp3")
        for path in (clip, audio):
            if not os.path.isfile(path):
                self.skipTest(f"Proof assets not present: {path}")
        out = os.path.join(MARS, "muxed", "test_mux.mp4")
        try:
            ff.mux_scene(clip, audio, out)
            dur = ff.probe_duration(out, "ffprobe")
            audio_dur = ff.probe_duration(audio, "ffprobe")
            # output ~= narration length (trimmed overshoot < 0.5s)
            self.assertGreater(dur, 0)
            self.assertLess(abs(dur - audio_dur), 0.5)
            # finite file: narration-length video at 720p must be small
            self.assertLess(os.path.getsize(out), 100 * 1024 * 1024)
        finally:
            if os.path.isfile(out):
                os.unlink(out)

    def test_mux_copy_video_mode(self):
        # Regression: main.py stage 5 calls mux_scene(copy_video=True).
        clip = os.path.join(MARS, "videos", "scene_001.mp4")
        audio = os.path.join(MARS, "audio", "scene_001.mp3")
        for path in (clip, audio):
            if not os.path.isfile(path):
                self.skipTest(f"Proof assets not present: {path}")
        out = os.path.join(MARS, "muxed", "test_mux_copy.mp4")
        try:
            ff.mux_scene(clip, audio, out, copy_video=True)
            dur = ff.probe_duration(out, "ffprobe")
            audio_dur = ff.probe_duration(audio, "ffprobe")
            self.assertGreater(dur, 0)
            self.assertLess(abs(dur - audio_dur), 0.5)
            self.assertLess(os.path.getsize(out), 100 * 1024 * 1024)
        finally:
            if os.path.isfile(out):
                os.unlink(out)

    def test_concat_copy_mode(self):
        # Regression: main.py stage 5 calls concat_scenes(copy=True).
        clip = os.path.join(MARS, "videos", "scene_001.mp4")
        audio = os.path.join(MARS, "audio", "scene_001.mp3")
        for path in (clip, audio):
            if not os.path.isfile(path):
                self.skipTest(f"Proof assets not present: {path}")
        muxed = os.path.join(MARS, "muxed", "test_concat_copy.mp4")
        final = os.path.join(MARS, "muxed", "test_concat_copy_final.mp4")
        try:
            ff.mux_scene(clip, audio, muxed, copy_video=True)
            ff.concat_scenes([muxed, muxed], final, copy=True)
            dur = ff.probe_duration(final, "ffprobe")
            single = ff.probe_duration(muxed, "ffprobe")
            self.assertAlmostEqual(dur, 2 * single, delta=0.5)
        finally:
            for path in (muxed, final):
                if os.path.isfile(path):
                    os.unlink(path)


if __name__ == "__main__":
    unittest.main()
