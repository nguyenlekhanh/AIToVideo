"""Unit tests: Continuous Video Chain (planner + engine + CLI).

No GPU/network: Ollama transport (ollama._post), the video provider, and
ffmpeg execution are mocked or stubbed. The key invariant under test:
clip N+1 starts from clip N's last frame, never from the source image.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import continuous as ch
import main
import ollama as ol


def chain_response(n=3, durations=None, omit=None):
    durations = durations or [6] * n
    clips = []
    for i in range(n):
        clip = {
            "id": i + 1, "duration": durations[i],
            "start_state": f"Start state {i + 1}.",
            "action": f"Action beat {i + 1}.",
            "end_state": f"End state {i + 1}.",
            "video_prompt": f"Video motion {i + 1}.",
        }
        if omit:
            clip.pop(omit, None)
        clips.append(clip)
    return {"message": {"content": json.dumps({"clips": clips})}}


def make_board(n=3, durations=None):
    durations = durations or [6] * n
    return {
        "user_prompt": "A model walks.",
        "source_image": "/src/model.jpg",
        "planner_model": "qwen3:8b",
        "clips": [{
            "id": i + 1, "duration": durations[i],
            "start_state": f"Start {i + 1}.",
            "action": f"Action {i + 1}.",
            "end_state": f"End {i + 1}.",
            "video_prompt": f"Motion {i + 1}.",
        } for i in range(n)],
    }


class FakeProvider:
    """Stub VideoProvider: records requests, materializes tiny outputs."""

    def __init__(self, fail_on=()):
        self.requests = []
        self.fail_on = set(fail_on)

    def generate(self, request):
        self.requests.append(request)
        if len(self.requests) in self.fail_on:
            raise RuntimeError("backend exploded")
        Path(str(request.output_path)).write_bytes(b"fakevideo")
        return SimpleNamespace(path=Path(str(request.output_path)),
                               width=64, height=64)


def write_file(path, data=b"bytes"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return path


class ContinuousCliTest(unittest.TestCase):
    def base(self, **over):
        argv = ["--project", "runway1", "--continuous", "--image", "m.jpg",
                "A model walks."]
        for key, value in over.items():
            argv += [f"--{key}", str(value)]
        return main.parse_args(argv)

    def test_requires_image(self):
        args = main.parse_args(["--project", "r", "--continuous", "Walk."])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_image_needs_continuous(self):
        args = main.parse_args(["--project", "r", "--image", "m.jpg", "Walk."])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_clips_needs_continuous(self):
        args = main.parse_args(["--project", "r", "--clips", "3", "Walk."])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_duration_needs_continuous(self):
        args = main.parse_args(["--project", "r", "--duration", "20", "Walk."])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_conflicts(self):
        bad = [
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--resume", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--stage", "video", "--resume", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--keyframes", "kf", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--script", "s.txt", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--character-reference", "c.png", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--scenes", "3", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--stop-after", "image", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--clips", "0", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--duration", "0", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--clips", "3", "--duration", "100", "Walk."],
        ]
        for argv in bad:
            with self.subTest(argv=argv):
                self.assertIsNotNone(
                    main.validate_cli_combination(main.parse_args(argv)))

    def test_valid_combinations(self):
        ok = [
            ["--project", "r", "--continuous", "--image", "m.jpg", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--clips", "6", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--duration", "40", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--clips", "6", "--duration", "40", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--scene", "2", "Walk."],
            ["--project", "r", "--continuous", "--image", "m.jpg",
             "--stop-after", "storyboard", "Walk."],
        ]
        for argv in ok:
            with self.subTest(argv=argv):
                self.assertIsNone(
                    main.validate_cli_combination(main.parse_args(argv)))

    def test_scene_allowed_without_resume_in_chain(self):
        args = main.parse_args(["--project", "r", "--continuous",
                                "--image", "m.jpg", "--scene", "2", "Walk."])
        self.assertIsNone(main.validate_cli_combination(args))

    def test_scene_still_needs_resume_outside_chain(self):
        args = main.parse_args(["--project", "r", "--scene", "2"])
        self.assertIsNotNone(main.validate_cli_combination(args))


class ChainPlannerTest(unittest.TestCase):
    def test_exact_count_and_fields(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(4)) as post:
            board = ch.plan_chain("A model walks.", model="m", num_clips=4)
        self.assertEqual(post.call_count, 1)
        self.assertEqual([c["id"] for c in board["clips"]], [1, 2, 3, 4])
        for clip in board["clips"]:
            for field in ("start_state", "action", "end_state",
                          "video_prompt", "duration"):
                self.assertTrue(clip[field])

    def test_no_images_sent_to_qwen(self):
        seen = []

        def fake_post(url, payload, timeout):
            seen.append(payload)
            return chain_response(2)

        with patch.object(ol, "_post", side_effect=fake_post):
            ch.plan_chain("Walk.", model="m", num_clips=2)
        self.assertTrue(seen)
        for payload in seen:
            for message in payload["messages"]:
                self.assertNotIn("images", message)

    def test_fence_stripping(self):
        inner = json.dumps({"clips": [{
            "id": 1, "duration": 5, "start_state": "S.", "action": "A.",
            "end_state": "E.", "video_prompt": "V."}]})
        fenced = f"```json\n{inner}\n```"
        with patch.object(ol, "_post",
                           return_value={"message": {"content": fenced}}):
            board = ch.plan_chain("Walk.", model="m", num_clips=1 - 1 + 1)
        self.assertEqual(len(board["clips"]), 1)

    def test_malformed_retries_then_fails(self):
        bad = {"message": {"content": "not json at all"}}
        with patch.object(ol, "_post", return_value=bad) as post:
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                ch.plan_chain("Walk.", model="m", num_clips=2)
        self.assertEqual(post.call_count, 3)

    def test_wrong_count_fails(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(2)) as post:
            with self.assertRaisesRegex(RuntimeError, "Chain planning"):
                ch.plan_chain("Walk.", model="m", num_clips=5)
        self.assertEqual(post.call_count, 3)

    def test_duration_bounds_rejected(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(2, durations=[2, 30])):
            with self.assertRaisesRegex(RuntimeError, "Chain planning"):
                ch.plan_chain("Walk.", model="m", num_clips=2)

    def test_missing_state_rejected(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(1, omit="end_state")):
            with self.assertRaisesRegex(RuntimeError, "Chain planning"):
                ch.plan_chain("Walk.", model="m", num_clips=1)

    def test_auto_count_bounded(self):
        with patch.object(ol, "_post", return_value=chain_response(5)):
            board = ch.plan_chain("Walk.", model="m")
        self.assertTrue(2 <= len(board["clips"]) <= 8)

    def test_total_duration_respected(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(
                               3, durations=[6, 7, 7])):
            board = ch.plan_chain("Walk.", model="m", total_duration=20)
        self.assertEqual(sum(c["duration"] for c in board["clips"]), 20)

    def test_total_duration_miss_fails(self):
        with patch.object(ol, "_post",
                           return_value=chain_response(
                               3, durations=[3, 3, 3])):
            with self.assertRaisesRegex(RuntimeError, "Chain planning"):
                ch.plan_chain("Walk.", model="m", total_duration=60)


class ChainEngineTest(unittest.TestCase):
    def run_chain(self, tmp, board, provider, **kwargs):
        out_dir = os.path.join(tmp, "continuous")
        source = write_file(os.path.join(tmp, "model.jpg"), b"sourcedata")
        kwargs.setdefault("negative_prompt", "")
        kwargs.setdefault("width", 64)
        kwargs.setdefault("height", 64)
        kwargs.setdefault("seed", 1)
        kwargs.setdefault("ffmpeg_exe", "ffmpeg")
        with patch("continuous.ff.extract_last_frame",
                   side_effect=lambda v, d, executable="ffmpeg": write_file(
                       d, b"lastframe-of-" + os.path.basename(
                           os.path.dirname(v)).encode())), \
             patch("continuous.st.valid_video_file", return_value=True), \
             patch("continuous.st.valid_image_file", return_value=True):
            return ch.run_chain(out_dir=out_dir, board=board,
                                source_image=source, provider=provider,
                                **kwargs)

    def test_first_clip_uses_source_then_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            results = self.run_chain(tmp, make_board(3), provider)
            self.assertEqual(len(results), 3)
            reqs = provider.requests
            self.assertEqual(len(reqs), 3)
            out_dir = os.path.join(tmp, "continuous")
            # clip 1 starts from the source image (materialized as start.png
            # with identical bytes)
            src_hash = ch.file_hash(os.path.join(tmp, "model.jpg"))
            self.assertTrue(str(reqs[0].input_image).endswith(
                os.path.join("001", "start.png")))
            self.assertEqual(ch.file_hash(str(reqs[0].input_image)), src_hash)
            # clips 2/3 start from previous last frames (copied into their
            # own start.png), never from the source image
            for i in (2, 3):
                prev_last = os.path.join(
                    out_dir, f"{i - 1:03d}", "lastframe.png")
                self.assertTrue(str(reqs[i - 1].input_image).endswith(
                    os.path.join(f"{i:03d}", "start.png")))
                self.assertEqual(
                    ch.file_hash(str(reqs[i - 1].input_image)),
                    ch.file_hash(prev_last))
                self.assertNotIn("model.jpg", str(reqs[i - 1].input_image))

    def test_prompt_assignment_and_continuity_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"))
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                ch.run_chain(out_dir=out_dir, board=make_board(2),
                             source_image=source, provider=provider,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
            self.assertEqual(provider.requests[0].prompt, "Motion 1.")
            self.assertTrue(provider.requests[1].prompt.startswith(
                ch.CONTINUITY_BLOCK))
            self.assertIn("Motion 2.", provider.requests[1].prompt)
            for i in (1, 2):
                text = Path(out_dir, f"{i:03d}", "prompt.txt").read_text()
                self.assertIn(f"Motion {i}.", text)

    def test_extraction_after_every_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"))
            with patch("continuous.ff.extract_last_frame") as extr, \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                extr.side_effect = lambda v, d, **k: write_file(d)
                ch.run_chain(out_dir=out_dir, board=make_board(3),
                             source_image=source, provider=provider,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
            self.assertEqual(extr.call_count, 3)
            got = sorted(os.path.basename(os.path.dirname(c.args[0]))
                         for c in extr.call_args_list)
            self.assertEqual(got, ["001", "002", "003"])

    def test_missing_previous_lastframe_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"))
            real_extract = lambda v, d, **k: write_file(d)  # noqa: E731
            calls = {"n": 0}

            def image_gate(path):
                # source check + clip 1 pre/post lastframe checks pass;
                # the clip 2 start check (previous lastframe) fails.
                calls["n"] += 1
                return calls["n"] <= 3

            with patch("continuous.ff.extract_last_frame",
                        side_effect=real_extract), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file",
                        side_effect=image_gate):
                with self.assertRaisesRegex(ValueError,
                                            "previous last frame"):
                    ch.run_chain(out_dir=out_dir, board=make_board(2),
                                 source_image=source, provider=provider,
                                 negative_prompt="", width=64, height=64,
                                 seed=1, ffmpeg_exe="ffmpeg")

    def test_resume_skips_valid_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"), b"src")
            # Consistent chain bytes: each start matches the previous
            # lastframe (clip 1 matches the source).
            blobs = {1: (b"src", b"last1"), 2: (b"last1", b"last2"),
                     3: (b"last2", b"last3")}
            manifest = {}
            for i in (1, 2, 3):
                start = write_file(
                    os.path.join(out_dir, f"{i:03d}", "start.png"),
                    blobs[i][0])
                write_file(os.path.join(out_dir, f"{i:03d}", "video.mp4"))
                write_file(os.path.join(out_dir, f"{i:03d}", "lastframe.png"),
                           blobs[i][1])
                manifest[str(i)] = {
                    "start_frame": start,
                    "start_frame_hash": ch.file_hash(start)}
            ch.save_manifest(os.path.join(out_dir, "manifest.json"),
                             manifest)
            provider = FakeProvider()
            with patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                results = ch.run_chain(
                    out_dir=out_dir, board=board, source_image=source,
                    provider=provider, negative_prompt="", width=64,
                    height=64, seed=1, ffmpeg_exe="ffmpeg")
            self.assertEqual(provider.requests, [])
            self.assertEqual(len(results), 3)

    def test_partial_resume_uses_previous_lastframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"), b"src")
            # clips 1-2 complete with matching manifest hashes
            manifest = {}
            prev = None
            for i in (1, 2):
                start_src = source if i == 1 else os.path.join(
                    out_dir, "001" if i == 2 else "000", "lastframe.png")
                start = write_file(os.path.join(out_dir, f"{i:03d}",
                                                "start.png"),
                                   Path(start_src).read_bytes())
                write_file(os.path.join(out_dir, f"{i:03d}", "video.mp4"))
                last = write_file(
                    os.path.join(out_dir, f"{i:03d}", "lastframe.png"),
                    f"last{i}".encode())
                manifest[str(i)] = {"start_frame": start,
                                    "start_frame_hash": ch.file_hash(start)}
                prev = last
            ch.save_manifest(os.path.join(out_dir, "manifest.json"),
                             manifest)
            provider = FakeProvider()
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                results = ch.run_chain(
                    out_dir=out_dir, board=board, source_image=source,
                    provider=provider, negative_prompt="", width=64,
                    height=64, seed=1, ffmpeg_exe="ffmpeg")
            self.assertEqual(len(provider.requests), 1)
            self.assertTrue(
                str(provider.requests[0].input_image).endswith(
                    os.path.join("003", "start.png")))
            self.assertEqual(
                ch.file_hash(str(provider.requests[0].input_image)),
                ch.file_hash(os.path.join(out_dir, "002", "lastframe.png")))
            self.assertEqual(len(results), 3)

    def test_stale_downstream_cascades(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"), b"src")
            provider = FakeProvider()
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(
                            d, b"frame" + os.path.basename(
                                os.path.dirname(v)).encode())), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                ch.run_chain(out_dir=out_dir, board=board,
                             source_image=source, provider=provider,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
            # corrupt clip 2's lastframe on disk (simulates a regen changing
            # it): clip 2 itself stays valid (its start is unchanged), but
            # clip 3's start goes stale and must regenerate (cascade)
            write_file(os.path.join(out_dir, "002", "lastframe.png"),
                       b"CHANGED")
            provider2 = FakeProvider()
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                ch.run_chain(out_dir=out_dir, board=board,
                             source_image=source, provider=provider2,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
            # clip 1 reused (hash matches), clip 2 reused (its own start
            # is unchanged), clip 3 regenerates from the changed lastframe
            self.assertEqual(len(provider2.requests), 1)
            self.assertTrue(
                str(provider2.requests[0].input_image).endswith(
                    os.path.join("003", "start.png")))
            self.assertEqual(
                ch.file_hash(str(provider2.requests[0].input_image)),
                ch.file_hash(os.path.join(out_dir, "002", "lastframe.png")))

    def test_scene_regenerates_only_that_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"), b"src")
            provider = FakeProvider()
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                # first full run to materialize lastframes
                ch.run_chain(out_dir=out_dir, board=board,
                             source_image=source, provider=provider,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
                provider2 = FakeProvider()
                results = ch.run_chain(
                    out_dir=out_dir, board=board, source_image=source,
                    provider=provider2, negative_prompt="", width=64,
                    height=64, seed=1, ffmpeg_exe="ffmpeg", only_clip=3)
            self.assertEqual(len(provider2.requests), 1)
            self.assertTrue(
                str(provider2.requests[0].input_image).endswith(
                    os.path.join("003", "start.png")))
            self.assertEqual(list(results), [3])

    def test_scene_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "out of range"):
                ch.run_chain(out_dir=os.path.join(tmp, "c"),
                             board=make_board(2), source_image="s",
                             provider=FakeProvider(), negative_prompt="",
                             width=1, height=1, seed=1, ffmpeg_exe="ffmpeg",
                             only_clip=9)

    def test_fixed_duration_backend_skips_duration_check(self):
        seen = []

        def fake_valid(path, expected=None):
            seen.append(expected)
            return True

        class FixedProvider(FakeProvider):
            fixed_duration = True

        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(2)
            out_dir = os.path.join(tmp, "continuous")
            source = write_file(os.path.join(tmp, "model.jpg"))
            provider = FixedProvider()
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file",
                        side_effect=fake_valid), \
                 patch("continuous.st.valid_image_file", return_value=True):
                ch.run_chain(out_dir=out_dir, board=board,
                             source_image=source, provider=provider,
                             negative_prompt="", width=64, height=64, seed=1,
                             ffmpeg_exe="ffmpeg")
            self.assertTrue(seen)
            self.assertTrue(all(e is None for e in seen))

    def test_backend_failure_names_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider(fail_on={2})
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, **k: write_file(d)), \
                 patch("continuous.st.valid_video_file", return_value=True), \
                 patch("continuous.st.valid_image_file", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "backend exploded"):
                    self.run_chain(tmp, make_board(3), provider)
            # clip 1 outputs remain for resume; no half-written clip 2 video
            self.assertTrue(os.path.isfile(
                os.path.join(tmp, "continuous", "001", "video.mp4")))
            self.assertFalse(os.path.isfile(
                os.path.join(tmp, "continuous", "002", "video.mp4")))


class ChainConcatTest(unittest.TestCase):
    def test_concat_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            board = make_board(3)
            seen = {}

            def fake_concat(paths, dest, executable="ffmpeg"):
                seen["paths"] = list(paths)
                seen["dest"] = dest
                write_file(dest)
                return dest

            with patch("continuous.ff.concat_scenes",
                        side_effect=fake_concat), \
                 patch("continuous.st.valid_video_file", return_value=True):
                final = ch.concat_chain(out_dir, board["clips"], "ffmpeg")
            self.assertEqual(
                seen["paths"],
                [os.path.join(out_dir, f"{i:03d}", "video.mp4")
                 for i in (1, 2, 3)])
            self.assertEqual(final, os.path.join(out_dir, "final.mp4"))

    def test_concat_missing_clip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            with patch("continuous.st.valid_video_file", return_value=False):
                with self.assertRaisesRegex(ValueError, "missing or invalid"):
                    ch.concat_chain(out_dir, make_board(2)["clips"],
                                    "ffmpeg")

    def test_board_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "board.json")
            ch.save_board(make_board(2), path)
            loaded = ch.load_board(path)
        self.assertEqual(len(loaded["clips"]), 2)

    def test_board_corrupt_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "board.json")
            Path(path).write_text("{nope")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                ch.load_board(path)
        with self.assertRaisesRegex(ValueError, "not found"):
            ch.load_board(os.path.join(tmp, "missing.json"))


class BoardSelectClipsTest(unittest.TestCase):
    def test_none_returns_board(self):
        board = make_board(3)
        self.assertIs(ch.select_clips(board, None), board)

    def test_equal_count_returns_board(self):
        board = make_board(3)
        self.assertIs(ch.select_clips(board, 3), board)

    def test_fewer_clips_slices_without_mutating(self):
        board = make_board(3)
        sliced = ch.select_clips(board, 2)
        self.assertEqual([c["id"] for c in sliced["clips"]], [1, 2])
        self.assertEqual(len(board["clips"]), 3)

    def test_more_clips_fails_clearly(self):
        with self.assertRaisesRegex(
                ValueError, r"contains 3 clips, but --clips 5 was requested"):
            ch.select_clips(make_board(3), 5)


class BoardReuseTest(unittest.TestCase):
    """Existing continuous_storyboard.json is authoritative: planner never
    runs, CLI prompt is ignored for planning, --clips is enforced."""

    def _setup(self, tmp, board=None):
        img = os.path.join(tmp, "model.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        cfgdir = os.path.join(tmp, "config")
        os.makedirs(cfgdir)
        shutil.copy(os.path.join(main.APP_DIR, "config", "models.json"),
                     os.path.join(cfgdir, "models.json"))
        if board is not None:
            projdir = os.path.join(tmp, "projects", "reuse1")
            os.makedirs(projdir)
            ch.save_board(board, os.path.join(
                projdir, "continuous_storyboard.json"))
        return img

    def _run(self, tmp, img, extra):
        argv = ["--project", "reuse1", "--continuous", "--image", img,
                "--stop-after", "storyboard"] + extra
        args = main.parse_args(argv)
        with patch.object(main, "APP_DIR", tmp):
            return main.run_continuous_command(args, {"ollama": {}})

    def _board_bytes(self, tmp):
        with open(os.path.join(tmp, "projects", "reuse1",
                               "continuous_storyboard.json"), "rb") as f:
            return f.read()

    def test_a_no_board_plans_and_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = self._setup(tmp)
            planned = make_board(3)
            planned["user_prompt"] = "Fresh prompt."
            with patch.object(ch, "plan_chain",
                               return_value=planned) as mock_plan:
                rc = self._run(tmp, img, ["Fresh prompt."])
            self.assertEqual(rc, 0)
            mock_plan.assert_called_once()
            saved = json.loads(self._board_bytes(tmp).decode("utf-8"))
            self.assertEqual(len(saved["clips"]), 3)
            self.assertEqual(saved["user_prompt"], "Fresh prompt.")

    def test_b_same_prompt_reuses_without_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            board["user_prompt"] = "Same prompt."
            img = self._setup(tmp, board)
            before = self._board_bytes(tmp)
            with patch.object(ch, "plan_chain",
                               side_effect=AssertionError("planner ran")):
                rc = self._run(tmp, img, ["Same prompt."])
            self.assertEqual(rc, 0)
            self.assertEqual(self._board_bytes(tmp), before)

    def test_c_different_prompt_reuses_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(3)
            board["user_prompt"] = "Original prompt."
            img = self._setup(tmp, board)
            before = self._board_bytes(tmp)
            with patch.object(ch, "plan_chain",
                               side_effect=AssertionError("planner ran")):
                rc = self._run(tmp, img, ["Something completely different."])
            self.assertEqual(rc, 0)
            self.assertEqual(self._board_bytes(tmp), before)

    def test_d_edited_board_used_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(2)
            board["user_prompt"] = "Original prompt."
            board["clips"][0]["action"] = "EDITED MARKER ACTION"
            img = self._setup(tmp, board)
            with patch.object(ch, "plan_chain",
                               side_effect=AssertionError("planner ran")):
                rc = self._run(tmp, img, ["Whatever."])
            self.assertEqual(rc, 0)
            saved = json.loads(self._board_bytes(tmp).decode("utf-8"))
            self.assertEqual(saved["clips"][0]["action"],
                             "EDITED MARKER ACTION")

    def test_e_too_many_clips_fails_clearly(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp:
            img = self._setup(tmp, make_board(3))
            with patch.object(ch, "plan_chain",
                               side_effect=AssertionError("planner ran")):
                err = io.StringIO()
                with redirect_stderr(err):
                    rc = self._run(tmp, img, ["Whatever.", "--clips", "5"])
            self.assertEqual(rc, 1)
            self.assertIn("contains 3 clips, but --clips 5 was requested",
                          err.getvalue())

    def test_fewer_clips_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = self._setup(tmp, make_board(3))
            with patch.object(ch, "plan_chain",
                               side_effect=AssertionError("planner ran")):
                rc = self._run(tmp, img, ["Whatever.", "--clips", "2"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
