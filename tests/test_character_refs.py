"""Unit tests: project-local character references for the Continuous Chain.

No GPU/network: Ollama transport (ollama._post), the video provider, and
ffmpeg execution are mocked or stubbed. Covers discovery, identity v2
merge/hash semantics, per-scene resolution, prompt composition,
resume/stale behavior, and backward compatibility.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import continuous as ch
import ollama as ol

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
PNG2 = b"\x89PNG\r\n\x1a\n" + b"1" * 64


class FakeProvider:
    """Stub VideoProvider: records requests, materializes tiny outputs."""

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        Path(str(request.output_path)).write_bytes(b"fakevideo")
        return SimpleNamespace(path=Path(str(request.output_path)),
                               width=64, height=64)


def write_file(path, data=b"bytes"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return path


def make_board(n=3, characters=None):
    board = {
        "user_prompt": "A model walks.",
        "source_image": "/src/model.jpg",
        "planner_model": "qwen3:8b",
        "clips": [{
            "id": i + 1, "duration": 6,
            "start_state": f"Start {i + 1}.",
            "action": f"Action {i + 1}.",
            "end_state": f"End {i + 1}.",
            "video_prompt": f"Motion {i + 1}.",
            "characters": [],
        } for i in range(n)],
    }
    if characters:
        for clip, ids in zip(board["clips"], characters):
            clip["characters"] = list(ids)
    return board


def identity_response(**over):
    body = {"identity": {
        "type": "human", "role": "", "age_appearance": "",
        "face": {"shape": "", "skin": "", "eyes": "", "nose": "",
                 "facial_hair": ""},
        "hair": "", "body": "", "clothing": "", "accessories": [],
        "color_palette": "", "visual_style": ""}}
    body["identity"].update(over)
    return {"message": {"content": json.dumps(body)}}


def setup_refs(tmp, files):
    ref_dir = os.path.join(tmp, "ImageCharacterReference")
    os.makedirs(ref_dir, exist_ok=True)
    for name, data in files.items():
        Path(ref_dir, name).write_bytes(data)
    return ref_dir


def ensure_no_qwen(tmp, board, **kwargs):
    """ensure_identity with Qwen forbidden (loads existing identity)."""
    out_dir = os.path.join(tmp, "continuous")
    with patch.object(ol, "_post",
                       side_effect=AssertionError("Qwen called")):
        return ch.ensure_identity(out_dir, "prompt", board, model="m",
                                  base_dir=tmp, **kwargs)


def run_locked_refs(tmp, board, provider, identity):
    out_dir = os.path.join(tmp, "continuous")
    source = write_file(os.path.join(tmp, "model.jpg"), b"sourcedata")
    with patch("continuous.ff.extract_last_frame",
               side_effect=lambda v, d, executable="ffmpeg": write_file(d)), \
         patch("continuous.st.valid_video_file", return_value=True), \
         patch("continuous.st.valid_image_file", return_value=True):
        results = ch.run_chain(out_dir=out_dir, board=board,
                               source_image=source, provider=provider,
                               identity=identity, negative_prompt="",
                               width=64, height=64, seed=1,
                               ffmpeg_exe="ffmpeg")
    return results, out_dir


class DiscoveryTest(unittest.TestCase):
    def test_discovers_project_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG,
                             "Character_002.jpg": PNG})
            found = ch.discover_character_refs(tmp)
            self.assertEqual([f["id"] for f in found],
                             ["Character_001", "Character_002"])
            self.assertEqual(found[0]["reference_image"],
                             "ImageCharacterReference/Character_001.png")
            self.assertTrue(os.path.isfile(found[0]["abspath"]))

    def test_natural_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_010.webp": PNG,
                             "Character_002.jpg": PNG,
                             "Character_001.png": PNG})
            found = ch.discover_character_refs(tmp)
            self.assertEqual([f["id"] for f in found],
                             ["Character_001", "Character_002",
                              "Character_010"])

    def test_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"character_003.JPG": PNG,
                             "CHARACTER_004.WebP": PNG})
            found = ch.discover_character_refs(tmp)
            self.assertEqual([f["id"] for f in found],
                             ["Character_003", "Character_004"])

    def test_ignores_unrelated(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"notes.txt": b"x", "random.png": PNG,
                             "Character_001.bmp": PNG,
                             "Character_abc.png": PNG,
                             "Character_001.png": PNG})
            found = ch.discover_character_refs(tmp)
            self.assertEqual([f["id"] for f in found], ["Character_001"])

    def test_missing_dir_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ch.discover_character_refs(tmp), [])

    def test_empty_dir_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "ImageCharacterReference"))
            self.assertEqual(ch.discover_character_refs(tmp), [])

    def test_does_not_search_global(self):
        # Only <base>/ImageCharacterReference is ever read: a base without
        # one yields [] even though decoys exist elsewhere.
        with tempfile.TemporaryDirectory() as tmp:
            decoy = os.path.join(tmp, "elsewhere")
            os.makedirs(decoy)
            Path(decoy, "Character_001.png").write_bytes(PNG)
            self.assertEqual(ch.discover_character_refs(tmp), [])

    def test_duplicate_id_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG,
                             "Character_001.jpg": PNG})
            with self.assertRaisesRegex(ValueError, "Duplicate character"):
                ch.discover_character_refs(tmp)


class IdentityDocTest(unittest.TestCase):
    def test_v2_shape_on_ensure(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            doc = json.load(open(os.path.join(
                tmp, "continuous", "identity.json"), encoding="utf-8"))
            self.assertEqual(doc["version"], 2)
            self.assertEqual(len(doc["characters"]), 1)
            entry = doc["characters"][0]
            self.assertEqual(entry["id"], "Character_001")
            self.assertEqual(entry["reference_image"],
                             "ImageCharacterReference/Character_001.png")
            self.assertEqual(entry["reference_sha256"], ch.file_hash(
                os.path.join(tmp, "ImageCharacterReference",
                             "Character_001.png")))
            self.assertIn("identity", entry)
            self.assertIn("identity_lock", entry)
            # legacy single profile still present
            self.assertIn("identity", doc)
            self.assertNotIn(os.path.join(tmp, "x"),
                             json.dumps(doc))

    def test_v1_shape_without_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            doc = json.load(open(os.path.join(
                tmp, "continuous", "identity.json"), encoding="utf-8"))
            self.assertEqual(doc["version"], 1)
            self.assertNotIn("characters", doc)

    def test_old_v1_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "continuous")
            os.makedirs(out_dir)
            Path(out_dir, "identity.json").write_text(json.dumps(
                {"version": 1, "identity": {"type": "human"},
                 "identity_lock": []}))
            doc = ch.load_identity_doc(os.path.join(out_dir, "identity.json"))
            self.assertEqual(doc["characters"], [])
            self.assertEqual(doc["identity"]["type"], "human")
            # single-profile loader unchanged
            self.assertEqual(
                ch.load_identity(os.path.join(out_dir, "identity.json"))["type"],
                "human")

    def test_merge_keeps_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            # user enriches the persisted profile, then adds a character
            doc_path = os.path.join(tmp, "continuous", "identity.json")
            doc = json.load(open(doc_path, encoding="utf-8"))
            doc["characters"][0]["identity"]["role"] = "shepherd"
            json.dump(doc, open(doc_path, "w"), indent=2)
            setup_refs(tmp, {"Character_002.jpg": PNG2})
            ensure_no_qwen(tmp, make_board(1))
            doc = json.load(open(doc_path, encoding="utf-8"))
            by_id = {c["id"]: c for c in doc["characters"]}
            self.assertEqual(by_id["Character_001"]["identity"]["role"],
                             "shepherd")
            self.assertEqual(by_id["Character_002"]["identity"]["role"], "")

    def test_removed_file_drops_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG,
                             "Character_002.jpg": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            os.unlink(os.path.join(tmp, "ImageCharacterReference",
                                   "Character_002.jpg"))
            ensure_no_qwen(tmp, make_board(1))
            doc = json.load(open(os.path.join(
                tmp, "continuous", "identity.json"), encoding="utf-8"))
            self.assertEqual([c["id"] for c in doc["characters"]],
                             ["Character_001"])

    def test_no_qwen_when_loading_with_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            ensure_no_qwen(tmp, make_board(1))


class HashingTest(unittest.TestCase):
    def test_reference_change_moves_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            before = ch.characters_hash(ch.load_chain_characters(tmp))
            Path(tmp, "ImageCharacterReference",
                 "Character_001.png").write_bytes(PNG2)
            after = ch.characters_hash(ch.load_chain_characters(tmp))
            self.assertNotEqual(before, after)

    def test_hash_stable_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ch.ensure_identity(os.path.join(tmp, "continuous"),
                                   "prompt", make_board(1), model="m",
                                   base_dir=tmp)
            before = ch.characters_hash(ch.load_chain_characters(tmp))
            Path(tmp, "ImageCharacterReference",
                 "notes.txt").write_text("hello")
            Path(tmp, "other.png").write_bytes(PNG2)
            after = ch.characters_hash(ch.load_chain_characters(tmp))
            self.assertEqual(before, after)

    def test_hash_portable_across_roots(self):
        with tempfile.TemporaryDirectory() as tmp1, \
                tempfile.TemporaryDirectory() as tmp2:
            setup_refs(tmp1, {"Character_001.png": PNG})
            setup_refs(tmp2, {"Character_001.png": PNG})
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                for tmp in (tmp1, tmp2):
                    ch.ensure_identity(os.path.join(tmp, "continuous"),
                                       "prompt", make_board(1), model="m",
                                       base_dir=tmp)
            self.assertEqual(
                ch.characters_hash(ch.load_chain_characters(tmp1)),
                ch.characters_hash(ch.load_chain_characters(tmp2)))

    def test_empty_characters_hash_stable(self):
        self.assertEqual(ch.characters_hash([]), ch.characters_hash([]))


class BoardCharactersTest(unittest.TestCase):
    def test_preserved(self):
        board = make_board(2, characters=[["Character_001"],
                                          ["Character_001", "Character_002"]])
        validated = ch._validate_chain(
            {"clips": board["clips"]}, 2, None)
        self.assertEqual(validated["clips"][0]["characters"],
                         ["Character_001"])
        self.assertEqual(validated["clips"][1]["characters"],
                         ["Character_001", "Character_002"])

    def test_missing_defaults_empty(self):
        raw = {"clips": [{"id": 1, "duration": 5, "start_state": "S.",
                          "action": "A.", "end_state": "E.",
                          "video_prompt": "V."}]}
        validated = ch._validate_chain(raw, 1, None)
        self.assertEqual(validated["clips"][0]["characters"], [])

    def test_invalid_rejected(self):
        raw = {"clips": [{"id": 1, "duration": 5, "start_state": "S.",
                          "action": "A.", "end_state": "E.",
                          "video_prompt": "V.", "characters": "C001"}]}
        with self.assertRaisesRegex(ValueError, "characters"):
            ch._validate_chain(raw, 1, None)


class PromptCharactersTest(unittest.TestCase):
    IDENT = {"type": "human", "role": "shepherd", "clothing": "white robes",
             "accessories": ["staff"]}

    def _chars(self):
        return [{"id": "Character_001",
                 "reference_image": "ImageCharacterReference/C001.png",
                 "reference_sha256": "x",
                 "identity": dict(self.IDENT),
                 "identity_lock": []},
                {"id": "Character_002",
                 "reference_image": "ImageCharacterReference/C002.png",
                 "reference_sha256": "y",
                 "identity": ch.default_identity(),
                 "identity_lock": []}]

    def test_context_only_listed_characters(self):
        board = make_board(2)
        mem = ch.snapshot_after(board, ch.initial_memory(board), 1)
        prompt = ch.build_video_prompt(board["clips"][1], self.IDENT, mem,
                                       [self._chars()[0]])
        self.assertIn("CHARACTER REFERENCE:", prompt)
        self.assertIn("Character_001", prompt)
        self.assertIn("shepherd", prompt)
        self.assertNotIn("Character_002", prompt)
        # full section order intact
        order = ["IDENTITY LOCK:", "CHARACTER REFERENCE:",
                 "CONTINUITY STATE", "CURRENT ACTION:", "END STATE:",
                 "CAMERA:", "CONTINUATION:"]
        positions = [prompt.index(section) for section in order]
        self.assertEqual(positions, sorted(positions))

    def test_no_section_without_characters(self):
        board = make_board(2)
        mem = ch.snapshot_after(board, ch.initial_memory(board), 1)
        prompt = ch.build_video_prompt(board["clips"][1], self.IDENT, mem,
                                       [])
        self.assertNotIn("CHARACTER REFERENCE:", prompt)

    def test_compact(self):
        board = make_board(1)
        mem = ch.snapshot_after(board, ch.initial_memory(board), 0)
        prompt = ch.build_video_prompt(board["clips"][0], self.IDENT, mem,
                                       self._chars())
        self.assertLess(len(prompt), 2000)
        self.assertNotIn("reference_sha256", prompt)
        self.assertNotIn("ImageCharacterReference/C001.png", prompt)

    def test_legacy_untouched(self):
        clip = make_board(2)["clips"][1]
        self.assertTrue(ch.build_video_prompt(clip).startswith(
            ch.CONTINUITY_BLOCK))


class ChainCharactersTest(unittest.TestCase):
    def _setup(self, tmp, ref_files, clip_chars):
        setup_refs(tmp, ref_files)
        board = make_board(len(clip_chars), characters=clip_chars)
        with patch.object(ol, "_post",
                           return_value=identity_response()):
            ident = ch.ensure_identity(os.path.join(tmp, "continuous"),
                                       "prompt", board, model="m",
                                       base_dir=tmp)
        return board, ident

    def test_per_scene_resolution_and_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(
                tmp, {"Character_001.png": PNG, "Character_002.jpg": PNG2},
                [["Character_001"], ["Character_001", "Character_002"],
                 ["Character_002"]])
            provider = FakeProvider()
            (_, out_dir) = run_locked_refs(tmp, board, provider, ident)
            self.assertEqual(len(provider.requests), 3)
            abspaths = [str(r.identity_reference)
                        for r in provider.requests]
            self.assertTrue(
                abspaths[0].endswith(os.path.join("ImageCharacterReference",
                                                 "Character_001.png")))
            self.assertTrue(
                abspaths[2].endswith(os.path.join("ImageCharacterReference",
                                                 "Character_002.jpg")))
            t1 = Path(out_dir, "001", "prompt.txt").read_text()
            self.assertIn("Character_001", t1)
            self.assertNotIn("Character_002", t1)

    def test_unknown_character_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(tmp, {"Character_001.png": PNG},
                                       [["Character_009"]])
            provider = FakeProvider()
            with self.assertRaisesRegex(ValueError, "unknown character"):
                run_locked_refs(tmp, board, provider, ident)
            self.assertEqual(provider.requests, [])

    def test_no_characters_falls_back_to_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_refs(tmp, {"Character_001.png": PNG})
            board = make_board(1)
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ident = ch.ensure_identity(
                    os.path.join(tmp, "continuous"), "prompt", board,
                    model="m", base_dir=tmp)
            provider = FakeProvider()
            run_locked_refs(tmp, board, provider, ident)
            self.assertTrue(str(provider.requests[0].identity_reference
                                ).endswith("model.jpg"))

    def test_manifest_hashes_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(tmp, {"Character_001.png": PNG},
                                       [["Character_001"]])
            provider = FakeProvider()
            (_, out_dir) = run_locked_refs(tmp, board, provider, ident)
            manifest = json.load(open(os.path.join(out_dir, "manifest.json"),
                                      encoding="utf-8"))
            self.assertIn("characters_hash", manifest["1"])
            chash = manifest["1"]["characters_hash"]
            self.assertEqual(chash, ch.characters_hash(
                ch.load_chain_characters(tmp)))
            provider2 = FakeProvider()
            (results, _) = run_locked_refs(tmp, board, provider2, ident)
            self.assertEqual(provider2.requests, [])
            self.assertEqual(len(results), 1)

    def test_stale_on_reference_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(
                tmp, {"Character_001.png": PNG, "Character_002.jpg": PNG2},
                [["Character_001"], ["Character_002"]])
            provider = FakeProvider()
            run_locked_refs(tmp, board, provider, ident)
            # swap only Character_001's pixels (same filename)
            Path(tmp, "ImageCharacterReference",
                 "Character_001.png").write_bytes(PNG2)
            provider2 = FakeProvider()
            (results, _) = run_locked_refs(tmp, board, provider2, ident)
            redone = sorted(
                int(Path(str(r.output_path)).parent.name)
                for r in provider2.requests)
            # clip 1 regenerates (its character changed); clip 2's start
            # frame then changes, so it cascades
            self.assertIn(1, redone)
            self.assertEqual(len(results), 2)

    def test_scene_uses_character(self):
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(
                tmp, {"Character_001.png": PNG, "Character_002.jpg": PNG2},
                [["Character_001"], ["Character_002"]])
            provider = FakeProvider()
            run_locked_refs(tmp, board, provider, ident)
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
                    out_dir=out_dir, board=board, source_image=source,
                    provider=provider2, identity=ident, negative_prompt="",
                    width=64, height=64, seed=1, ffmpeg_exe="ffmpeg",
                    only_clip=2)
            self.assertEqual(len(provider2.requests), 1)
            req = provider2.requests[0]
            self.assertTrue(str(req.identity_reference).endswith(
                os.path.join("ImageCharacterReference", "Character_002.jpg")))
            self.assertIn("Character_002",
                          Path(out_dir, "002", "prompt.txt").read_text())
            self.assertEqual(list(results), [2])

    def test_old_manifest_grandfathered(self):
        # entries without characters_hash (pre-upgrade runs) do not force
        # regeneration when references are unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            board, ident = self._setup(tmp, {"Character_001.png": PNG},
                                       [["Character_001"]])
            provider = FakeProvider()
            (_, out_dir) = run_locked_refs(tmp, board, provider, ident)
            manifest_path = os.path.join(out_dir, "manifest.json")
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            for entry in manifest.values():
                if isinstance(entry, dict):
                    entry.pop("characters_hash", None)
            json.dump(manifest, open(manifest_path, "w"), indent=2)
            provider2 = FakeProvider()
            (results, _) = run_locked_refs(tmp, board, provider2, ident)
            self.assertEqual(provider2.requests, [])
            self.assertEqual(len(results), 1)


class NoRefsRegressionTest(unittest.TestCase):
    def test_run_without_refs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = make_board(2)
            with patch.object(ol, "_post",
                               return_value=identity_response()):
                ident = ch.ensure_identity(
                    os.path.join(tmp, "continuous"), "prompt", board,
                    model="m", base_dir=tmp)
            provider = FakeProvider()
            (results, out_dir) = run_locked_refs(tmp, board, provider, ident)
            self.assertEqual(len(results), 2)
            manifest = json.load(open(os.path.join(out_dir, "manifest.json"),
                                      encoding="utf-8"))
            self.assertIn("characters_hash", manifest["1"])
            doc = json.load(open(os.path.join(out_dir, "identity.json"),
                                 encoding="utf-8"))
            self.assertEqual(doc["version"], 1)
            self.assertNotIn("characters", doc)


if __name__ == "__main__":
    unittest.main()
