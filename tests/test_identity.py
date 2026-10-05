"""Unit tests: Identity Lock + Continuity Memory for the Continuous Chain.

No GPU/network: Ollama transport (ollama._post), the video provider, and
ffmpeg execution are mocked or stubbed. Covers identity lifecycle,
memory lifecycle, composed prompts, chained execution with hashes,
resume/--scene behavior, and the --identity-file CLI option.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continuous as ch
import main
import ollama as ol
from tests.test_continuous import FakeProvider, make_board, write_file


def identity_response(**over):
    body = {"identity": {
        "type": "human", "role": "shepherd",
        "age_appearance": "", "face": {"shape": "", "skin": "",
                                       "eyes": "", "nose": "",
                                       "facial_hair": "beard"},
        "hair": "", "body": "", "clothing": "white robes",
        "accessories": ["staff"], "color_palette": "",
        "visual_style": ""}}
    body["identity"].update(over)
    return {"message": {"content": json.dumps(body)}}


def run_locked(tmp, board, provider, identity, **kwargs):
    out_dir = os.path.join(tmp, "continuous")
    source = write_file(os.path.join(tmp, "model.jpg"), b"sourcedata")
    kwargs.setdefault("negative_prompt", "")
    kwargs.setdefault("width", 64)
    kwargs.setdefault("height", 64)
    kwargs.setdefault("seed", 1)
    kwargs.setdefault("ffmpeg_exe", "ffmpeg")
    with patch("continuous.ff.extract_last_frame",
               side_effect=lambda v, d, executable="ffmpeg": write_file(d)), \
         patch("continuous.st.valid_video_file", return_value=True), \
         patch("continuous.st.valid_image_file", return_value=True):
        return ch.run_chain(out_dir=out_dir, board=board,
                            source_image=source, provider=provider,
                            identity=identity, **kwargs), out_dir


class IdentityLifecycleTest(unittest.TestCase):
    def test_create_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            with patch.object(ol, "_post",
                               return_value=identity_response()) as post:
                ident = ch.ensure_identity(
                    out_dir, "A shepherd walks.", make_board(2),
                    model="m")
            self.assertEqual(post.call_count, 1)
            self.assertTrue(os.path.isfile(
                os.path.join(out_dir, "identity.json")))
            self.assertEqual(ident["clothing"], "white robes")
            # Unknown fields stay empty (conservative, no hallucination).
            self.assertEqual(ident["face"]["eyes"], "")

    def test_second_run_loads_without_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                first = ch.ensure_identity(
                    out_dir, "A shepherd walks.", make_board(2), model="m")
            with patch.object(ol, "_post",
                               side_effect=AssertionError("Qwen called")):
                second = ch.ensure_identity(
                    out_dir, "A shepherd walks.", make_board(2), model="m")
            self.assertEqual(first, second)

    def test_identity_immutable_across_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = {"type": "human", "role": "shepherd"}
            out_dir = os.path.join(tmp, "continuous")
            ch.save_identity(ch.normalize_identity(ident), os.path.join(
                out_dir, "identity.json"))
            (results, _) = run_locked(tmp, make_board(3), provider,
                                      ch.normalize_identity(ident))
            before = Path(out_dir, "identity.json").read_bytes()
            self.assertEqual(len(results), 3)
            self.assertEqual(Path(out_dir, "identity.json").read_bytes(),
                             before)

    def test_hash_deterministic(self):
        a = {"type": "human", "clothing": "robes", "face": {"eyes": ""}}
        b = {"clothing": "robes", "type": "human", "face": {"eyes": ""},
             "extra": "dropped"}
        self.assertEqual(ch.identity_hash(a), ch.identity_hash(b))
        c = dict(a, clothing="armor")
        self.assertNotEqual(ch.identity_hash(a), ch.identity_hash(c))

    def test_identity_file_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = os.path.join(tmp, "custom.json")
            Path(custom).write_text(json.dumps(
                {"identity": {"type": "human", "role": "knight",
                              "clothing": "plate armor"}}))
            out_dir = os.path.join(tmp, "continuous")
            with patch.object(ol, "_post",
                               side_effect=AssertionError("Qwen called")):
                ident = ch.ensure_identity(
                    out_dir, "A knight fights.", make_board(1), model="m",
                    identity_file=custom)
            self.assertEqual(ident["role"], "knight")
            saved = json.load(open(os.path.join(out_dir, "identity.json")))
            self.assertEqual(saved["identity"]["role"], "knight")

    def test_identity_file_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                ch.ensure_identity(
                    os.path.join(tmp, "c"), "x", make_board(1), model="m",
                    identity_file=os.path.join(tmp, "nope.json"))

    def test_corrupt_identity_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            os.makedirs(out_dir)
            Path(out_dir, "identity.json").write_text("{broken")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                ch.ensure_identity(out_dir, "x", make_board(1), model="m")

    def test_malformed_qwen_retries_then_fails(self):
        bad = {"message": {"content": "not json"}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ol, "_post", return_value=bad) as post:
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    ch.ensure_identity(
                        os.path.join(tmp, "c"), "x", make_board(1), model="m")
        self.assertEqual(post.call_count, 3)

    def test_no_images_sent_for_identity(self):
        seen = []

        def fake_post(url, payload, timeout):
            seen.append(payload)
            return identity_response()

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ol, "_post", side_effect=fake_post):
                ch.ensure_identity(os.path.join(tmp, "c"), "x",
                                   make_board(1), model="m")
        for payload in seen:
            for message in payload["messages"]:
                self.assertNotIn("images", message)


class MemoryLifecycleTest(unittest.TestCase):
    def test_memory_created_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            out_dir = os.path.join(tmp, "continuous")
            ch.save_identity(ident, os.path.join(out_dir, "identity.json"))
            (_, _) = run_locked(tmp, make_board(3), provider, ident)
            mem = json.load(open(os.path.join(out_dir, "memory.json")))
            self.assertEqual(mem["last_completed_clip"], 3)
            self.assertEqual(len(mem["story_state"]["completed"]), 3)
            self.assertEqual(mem["story_state"]["current"], "End 3.")
            self.assertEqual(mem["story_state"]["next"], "")
            # memory and identity are distinct documents
            self.assertNotIn("identity_lock", mem)
            self.assertNotIn("story_state", json.load(
                open(os.path.join(out_dir, "identity.json"))))

    def test_memory_persists_across_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            (first, out_dir) = run_locked(tmp, make_board(3), provider,
                                          ident)
            snap = Path(out_dir, "memory.json").read_bytes()
            provider2 = FakeProvider()
            (second, _) = run_locked(tmp, make_board(3), provider2, ident)
            self.assertEqual(provider2.requests, [])
            self.assertEqual(len(second), 3)
            self.assertEqual(Path(out_dir, "memory.json").read_bytes(), snap)

    def test_missing_memory_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            (_, out_dir) = run_locked(tmp, make_board(2), provider, ident)
            os.unlink(os.path.join(out_dir, "memory.json"))
            provider2 = FakeProvider()
            (results, _) = run_locked(tmp, make_board(2), provider2, ident)
            self.assertEqual(provider2.requests, [])
            self.assertEqual(len(results), 2)
            mem = json.load(open(os.path.join(out_dir, "memory.json")))
            self.assertEqual(mem["last_completed_clip"], 2)

    def test_corrupt_memory_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            (_, out_dir) = run_locked(tmp, make_board(1), provider, ident)
            Path(out_dir, "memory.json").write_text("{broken")
            provider2 = FakeProvider()
            (results, _) = run_locked(tmp, make_board(1), provider2, ident)
            self.assertEqual(provider2.requests, [])
            self.assertEqual(len(results), 1)

    def test_memory_snapshot_chains_clips(self):
        board = make_board(3)
        base = ch.initial_memory(board)
        self.assertEqual(base["story_state"]["current"], "Action 1.")
        self.assertEqual(base["story_state"]["next"], "Action 2.")
        after2 = ch.snapshot_after(board, base, 2)
        self.assertEqual(after2["last_completed_clip"], 2)
        self.assertEqual(len(after2["story_state"]["completed"]), 2)
        self.assertEqual(after2["story_state"]["current"], "Action 3.")
        self.assertEqual(ch.snapshot_after(board, base, 0)["story_state"],
                         base["story_state"])


class LockedPromptTest(unittest.TestCase):
    IDENT = {"type": "human", "role": "shepherd", "clothing": "white robes",
             "accessories": ["staff"]}

    def test_clip1_initial_state_no_continuation(self):
        board = make_board(2)
        mem = ch.snapshot_after(board, ch.initial_memory(board), 0)
        prompt = ch.build_video_prompt(board["clips"][0], self.IDENT, mem)
        self.assertIn("IDENTITY LOCK:", prompt)
        self.assertIn("white robes", prompt)
        self.assertIn("INITIAL STATE:", prompt)
        self.assertIn("CURRENT ACTION:", prompt)
        self.assertIn("END STATE:", prompt)
        self.assertIn("CAMERA:", prompt)
        self.assertNotIn("CONTINUATION:", prompt)
        self.assertIn("Action 1.", prompt)

    def test_clip2_continues_with_memory(self):
        board = make_board(3)
        mem = ch.snapshot_after(board, ch.initial_memory(board), 1)
        prompt = ch.build_video_prompt(board["clips"][1], self.IDENT, mem)
        self.assertIn("IDENTITY LOCK:", prompt)
        self.assertIn("CONTINUITY STATE (after clip 1):", prompt)
        self.assertIn("Action 2.", prompt)
        self.assertIn("CONTINUATION:", prompt)
        self.assertIn("Do not restart the action.", prompt)

    def test_lock_lines_present_and_compact(self):
        prompt = ch.build_video_prompt(make_board(1)["clips"][0],
                                       self.IDENT, None)
        for line in ch.IDENTITY_LOCK_LINES:
            self.assertIn(line, prompt)
        # compact: no 500-word dump for a sparse profile
        self.assertLess(len(prompt), 2000)

    def test_legacy_format_unchanged_without_identity(self):
        clip = make_board(2)["clips"][1]
        self.assertTrue(ch.build_video_prompt(clip).startswith(
            ch.CONTINUITY_BLOCK))
        self.assertEqual(ch.build_video_prompt(make_board(2)["clips"][0]),
                         "Motion 1.")


class LockedChainTest(unittest.TestCase):
    def test_hashes_recorded_and_chain_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human", "role": "s"})
            (_, out_dir) = run_locked(tmp, make_board(3), provider, ident)
            manifest = json.load(
                open(os.path.join(out_dir, "manifest.json")))
            ihash = ch.identity_hash(ident)
            for i in ("1", "2", "3"):
                self.assertEqual(manifest[i]["identity_hash"], ihash)
                self.assertIn("memory_hash", manifest[i])
            mem = json.load(open(os.path.join(out_dir, "memory.json")))
            self.assertEqual(
                manifest["3"]["memory_hash"], ch.memory_hash(mem))
            # prompt.txt carries the lock for every clip
            for i in (1, 2, 3):
                text = Path(out_dir, f"{i:03d}", "prompt.txt").read_text()
                self.assertIn("IDENTITY LOCK:", text)

    def test_identity_change_marks_all_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            run_locked(tmp, make_board(2), provider, ident)
            other = ch.normalize_identity({"type": "human",
                                           "clothing": "armor"})
            provider2 = FakeProvider()
            (results, _) = run_locked(tmp, make_board(2), provider2, other)
            self.assertEqual(len(provider2.requests), 2)
            self.assertEqual(len(results), 2)

    def test_scene_uses_locked_state_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human", "role": "s"})
            run_locked(tmp, make_board(3), provider, ident)
            provider2 = FakeProvider()
            out_dir = os.path.join(tmp, "continuous")
            source = os.path.join(tmp, "model.jpg")
            with patch("continuous.ff.extract_last_frame",
                        side_effect=lambda v, d, executable="ffmpeg":
                        write_file(d)), \
                 patch("continuous.st.valid_video_file",
                        return_value=True), \
                 patch("continuous.st.valid_image_file",
                        return_value=True):
                results = ch.run_chain(
                    out_dir=out_dir, board=make_board(3),
                    source_image=source, provider=provider2,
                    negative_prompt="", width=64, height=64, seed=1,
                    ffmpeg_exe="ffmpeg", only_clip=3, identity=ident)
            self.assertEqual(len(provider2.requests), 1)
            req = provider2.requests[0]
            self.assertIn("IDENTITY LOCK:", req.prompt)
            # memory snapshot fed to clip 3 reflects clips 1-2 completed
            self.assertIn("End 1.", req.prompt)
            self.assertIn("End 2.", req.prompt)
            # manifest carries this run's identity hash
            manifest = json.load(
                open(os.path.join(out_dir, "manifest.json")))
            self.assertEqual(manifest["3"]["identity_hash"],
                             ch.identity_hash(ident))
            self.assertEqual(list(results), [3])

    def test_identity_reference_passed_to_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeProvider()
            ident = ch.normalize_identity({"type": "human"})
            run_locked(tmp, make_board(1), provider, ident)
            req = provider.requests[0]
            self.assertEqual(str(req.identity_reference),
                             os.path.join(tmp, "model.jpg"))


class IdentityCliTest(unittest.TestCase):
    def test_flag_parses(self):
        args = main.parse_args(["--project", "r", "--continuous",
                                "--image", "m.jpg", "--identity-file",
                                "id.json", "Walk."])
        self.assertEqual(args.identity_file, "id.json")

    def test_flag_needs_continuous(self):
        args = main.parse_args(["--project", "r", "--identity-file",
                                "id.json", "Walk."])
        self.assertIsNotNone(main.validate_cli_combination(args))

    def test_valid_combination(self):
        args = main.parse_args(["--project", "r", "--continuous",
                                "--image", "m.jpg", "--identity-file",
                                "id.json", "Walk."])
        self.assertIsNone(main.validate_cli_combination(args))


class PlannerStateTest(unittest.TestCase):
    def test_initial_state_preserved(self):
        body = {"initial_state": {"global": {"location": "runway"},
                                  "characters": [{"id": "model"}],
                                  "objects": [],
                                  "camera": {"shot": "wide"}},
                "clips": [{
                    "id": 1, "duration": 5, "start_state": "S.",
                    "action": "A.", "end_state": "E.", "video_prompt": "V.",
                    "state": {"global": {"location": "runway end"}}}]}
        with patch.object(ol, "_post", return_value={
                "message": {"content": __import__("json").dumps(body)}}):
            board = ch.plan_chain("Walk.", model="m", num_clips=1)
        self.assertEqual(board["initial_state"]["global"]["location"],
                         "runway")
        self.assertEqual(board["clips"][0]["state"]["global"]["location"],
                         "runway end")
        mem = ch.snapshot_after(board, ch.initial_memory(board), 1)
        self.assertEqual(mem["global"]["location"], "runway end")
        self.assertEqual(mem["characters"][0]["id"], "model")


if __name__ == "__main__":
    unittest.main()
