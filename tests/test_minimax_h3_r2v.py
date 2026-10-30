"""Unit tests: MiniMax H3 R2V backend + --character-reference-dir (no GPU)."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ffmpeg as ffmod
import main
import promptfiles as pf
import state as st
from providers.comfy import ComfyClient
from providers.errors import ProviderError
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.minimax_h3_r2v import MiniMaxH3R2VProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(
    AI_VIDEO_DIR, "workflows_template",
    "video_minimax_h3_r2v_with_H3_Memory_Optimize_H3_Sparse_Attention.json")


class StubClient(ComfyClient):
    """Captures queued workflows/uploads; writes no files to a server."""

    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.queued = []
        self.uploads = []

    def run(self, workflow, output_keys, dest_path: str) -> str:
        self.queued.append(workflow)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"fake")
        return dest_path

    def upload_image(self, image_path: str) -> str:
        self.uploads.append(str(image_path))
        return os.path.basename(str(image_path))


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "video", "minimax_h3_r2v")
    settings = dict(entry, **overrides)
    return MiniMaxH3R2VProvider(name="minimax_h3_r2v", settings=settings,
                                client=StubClient())


def make_images(tmp, *names):
    d = os.path.join(tmp, "chars")
    os.makedirs(d, exist_ok=True)
    for name in names:
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
    return d


def make_prompt_dir(tmp, durations, *texts):
    d = os.path.join(tmp, "prompt")
    os.makedirs(d, exist_ok=True)
    for i, text in enumerate(texts, start=1):
        with open(os.path.join(d, f"{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"clips": {str(i): {"duration": durations[i - 1]}
                             for i in range(1, len(texts) + 1)}}, f)
    return d


def make_ctx(tmp, provider, character_refs=None, image_1_dir=None,
             chain=False):
    state = st.new_state("w")
    return {"project": "w", "base_dir": tmp, "base_seed": 1,
            "state": state, "state_path": os.path.join(tmp, "s.json"),
            "img_dir": tmp, "vid_dir": tmp, "video_provider": provider,
            "video_model": "minimax_h3_r2v", "vid_neg": "",
            "target_w": 480, "target_h": 864, "default_duration": 5.0,
            "prompt_mode": True, "first_frame_chain": chain,
            "image_1_dir": image_1_dir, "character_refs": character_refs,
            "ffmpeg_exe": "ffmpeg"}


def fake_extract(video_path, out_path, executable="ffmpeg"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(b"png")
    return out_path


def run_stage(ctx, scenes, valid=True, **kwargs):
    if isinstance(valid, bool):
        vf = patch.object(st, "valid_video_file", return_value=valid)
    else:
        vf = patch.object(st, "valid_video_file", side_effect=valid)
    with vf, patch.object(st, "valid_image_file", return_value=False), \
         patch.object(ffmod, "extract_last_frame", side_effect=fake_extract):
        return main.run_video_stage(ctx, scenes, {}, **kwargs)


class ProviderStructureTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("video", AI_VIDEO_DIR, "minimax_h3_r2v",
                                   client=StubClient())
        self.assertIsInstance(provider, MiniMaxH3R2VProvider)
        self.assertEqual(provider.provider_id, "minimax_h3_r2v")
        self.assertFalse(provider.needs_input_image)
        self.assertTrue(provider.supports_first_frame)
        self.assertTrue(provider.supports_character_refs)
        self.assertTrue(provider.supports_product_refs)

    def test_workflow_preserves_memory_and_sparse_attention(self):
        wf = make_provider().load_workflow()
        self.assertEqual(len(wf), 32)
        classes = [n.get("class_type") for n in wf.values()]
        self.assertEqual(classes.count("H3MemoryOptimization"), 2)
        self.assertEqual(classes.count("H3SparseAttention"), 2)
        self.assertEqual(classes.count("MiniMaxH3ReferenceToVideo"), 1)
        self.assertEqual(classes.count("LoadImage"), 4)

    def test_exporter_reproduces_shipped_workflow(self):
        import export_minimax_h3_r2v as exporter
        self.assertEqual(exporter.build(), make_provider().load_workflow())

    def test_canonical_template_unchanged_by_export_and_runtime(self):
        with open(TEMPLATE, "rb") as f:
            before = f.read()
        import export_minimax_h3_r2v as exporter
        exporter.build()
        provider = make_provider()
        provider.generate(VideoRequest(
            prompt="p", negative_prompt="", input_image=None,
            width=768, height=512, duration=5, seed=1,
            output_path=Path(tempfile.gettempdir()) / "r2v_tpl_check.mp4",
            first_frame="guide.png"))
        with open(TEMPLATE, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_missing_workflow_is_clear_error(self):
        provider = make_provider(workflow_path=os.path.join(AI_VIDEO_DIR,
                                                            "nope.json"))
        with self.assertRaises(ProviderError) as ctx:
            provider.load_workflow()
        self.assertIn("minimax_h3_r2v", str(ctx.exception))

    def test_t2v_workflow_rejected(self):
        t2v = lookup(AI_VIDEO_DIR, "video", "minimax_h3")
        provider = make_provider(workflow_path=t2v["workflow_path"])
        with self.assertRaises(ProviderError):
            provider.load_workflow()


class ProviderSubstitutionTest(unittest.TestCase):
    def _customize(self, refs=None, first_frame=None):
        provider = make_provider()
        wf = provider.customize(
            provider.load_workflow(), video_prompt="P", seed=7, duration=5.0,
            aspect_label="16:9 (Widescreen)", megapixels=1.0,
            character_refs=refs, first_frame=first_frame)
        return provider, wf

    def test_character_ref_slots_follow_template_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "2.jpg", "3.webp")
            refs = [os.path.join(d, n) for n in ("1.png", "2.jpg", "3.webp")]
            _provider, wf = self._customize(refs=refs)
            gen = wf["136"]["inputs"]
            # Canonical mapping: 0->137, 1->139, 2->147, 3->153.
            self.assertEqual(wf["137"]["inputs"]["image"], "1.png")
            self.assertEqual(wf["139"]["inputs"]["image"], "2.jpg")
            self.assertEqual(wf["147"]["inputs"]["image"], "3.webp")
            self.assertIsNone(gen["ref_images.ref_image_3"])
            self.assertNotIn("ref_images.ref_image_4", gen)

    def test_product_refs_precede_character_refs_and_keep_guide_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "p1.png", "p2.png", "c1.png")
            provider = make_provider()
            wf = provider.customize(
                provider.load_workflow(), video_prompt="P", seed=7,
                duration=5.0, aspect_label="16:9 (Widescreen)",
                megapixels=1.0,
                product_refs=[os.path.join(d, "p1.png"),
                              os.path.join(d, "p2.png")],
                character_refs=[os.path.join(d, "c1.png")],
                first_frame="prev.png")
            gen = wf["136"]["inputs"]
            self.assertEqual(wf["137"]["inputs"]["image"], "p1.png")
            self.assertEqual(wf["139"]["inputs"]["image"], "p2.png")
            self.assertEqual(wf["147"]["inputs"]["image"], "c1.png")
            self.assertIsNone(gen["ref_images.ref_image_3"])
            self.assertEqual(gen["ref_images.ref_image_4"], ["1", 0])
            self.assertEqual(wf["1"]["inputs"], {"image": "prev.png"})

    def test_too_many_product_and_character_refs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "2.png", "3.png", "4.png", "5.png")
            provider = make_provider()
            with self.assertRaises(ValueError) as ctx:
                provider.customize(
                    provider.load_workflow(), video_prompt="P", seed=7,
                    duration=5.0, aspect_label="16:9 (Widescreen)",
                    megapixels=1.0,
                    product_refs=[os.path.join(d, n)
                                  for n in ("1.png", "2.png", "3.png")],
                    character_refs=[os.path.join(d, "4.png"),
                                    os.path.join(d, "5.png")])
            self.assertIn("Too many reference images", str(ctx.exception))

    def test_no_refs_disconnects_all_slots(self):
        _provider, wf = self._customize()
        gen = wf["136"]["inputs"]
        for i in range(4):
            self.assertIsNone(gen[f"ref_images.ref_image_{i}"])

    def test_guide_frame_uses_slot4_and_keeps_character_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png")
            ref = os.path.join(d, "1.png")
            _provider, wf = self._customize(refs=[ref],
                                           first_frame="prev_last.png")
            gen = wf["136"]["inputs"]
            self.assertEqual(wf["1"]["inputs"], {"image": "prev_last.png"})
            self.assertEqual(gen["ref_images.ref_image_4"], ["1", 0])
            self.assertEqual(wf["137"]["inputs"]["image"], "1.png")
            self.assertNotIn("last_frame", gen)

    def test_too_many_refs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "2.png", "3.png", "4.png", "5.png")
            refs = [os.path.join(d, f"{i}.png") for i in range(1, 6)]
            with self.assertRaises(ValueError) as ctx:
                self._customize(refs=refs)
            self.assertIn("at most 4", str(ctx.exception))

    def test_only_allowed_inputs_change(self):
        provider = make_provider()
        template = provider.load_workflow()
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png")
            after = provider.customize(
                template, video_prompt="P", seed=7, duration=5.0,
                aspect_label="16:9 (Widescreen)", megapixels=1.0,
                character_refs=[os.path.join(d, "1.png")],
                first_frame="prev.png")
        allowed = {"MiniMaxH3ReferenceToVideo", "RandomNoise", "PrimitiveFloat",
                   "ResolutionSelector", "PrimitiveStringMultiline",
                   "LoadImage"}
        for nid, node in template.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in {nid} ({node['class_type']})")
        for nid in ("124", "125", "127", "128", "133", "134", "135"):
            if nid in template:
                self.assertEqual(template[nid], after[nid])

    def test_template_file_unchanged_on_disk(self):
        provider = make_provider()
        with open(provider.workflow_path, "rb") as f:
            before = f.read()
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png")
            provider.customize(
                provider.load_workflow(), video_prompt="P", seed=7,
                duration=5.0, aspect_label="16:9 (Widescreen)", megapixels=1.0,
                character_refs=[os.path.join(d, "1.png")],
                first_frame="prev.png")
        with open(provider.workflow_path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_upload_cache_avoids_reuploading_same_reference(self):
        provider = make_provider()
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png")
            ref = os.path.join(d, "1.png")
            for i in range(2):
                provider.generate(VideoRequest(
                    prompt="p", negative_prompt="", input_image=None,
                    width=768, height=512, duration=5, seed=i,
                    output_path=Path(tmp) / f"o{i}.mp4",
                    character_refs=[ref]))
            self.assertEqual(provider.client.uploads.count(ref), 1)

    def test_input_image_rejected(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="", input_image=Path("x.png"),
                           width=768, height=512, duration=5, seed=1,
                           output_path=Path("x.mp4"))
        with self.assertRaises(ProviderError):
            provider.generate(req)

    def test_negative_prompt_rejected(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="blurry",
                           input_image=None, width=768, height=512,
                           duration=5, seed=1, output_path=Path("x.mp4"))
        with self.assertRaises(ProviderError):
            provider.generate(req)

    def test_duration_range(self):
        provider = make_provider()
        for bad in (0, 16, 99):
            with self.subTest(bad=bad):
                req = VideoRequest(prompt="p", negative_prompt="",
                                   input_image=None, width=768, height=512,
                                   duration=bad, seed=1,
                                   output_path=Path("x.mp4"))
                with self.assertRaises(ValueError):
                    provider.generate(req)


class CharacterRefDiscoveryTest(unittest.TestCase):
    def test_numeric_order_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "10.png", "2.JPG", "1.PNG", "3.webp")
            self.assertEqual(
                [os.path.basename(p) for p in pf.discover_character_refs(d)],
                ["1.PNG", "2.JPG", "3.webp", "10.png"])

    def test_ignores_unrelated_files_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "character.png", "notes.txt",
                            "4.gif")
            os.makedirs(os.path.join(d, "2.png"))
            self.assertEqual(
                [os.path.basename(p) for p in pf.discover_character_refs(d)],
                ["1.png"])

    def test_duplicate_numbers_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "1.jpg")
            with self.assertRaises(ValueError) as ctx:
                pf.discover_character_refs(d)
            self.assertIn("1.png", str(ctx.exception))
            self.assertIn("1.jpg", str(ctx.exception))

    def test_empty_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp)
            with self.assertRaises(ValueError) as ctx:
                pf.discover_character_refs(d)
            self.assertIn("No character reference images", str(ctx.exception))

    def test_missing_directory_fails(self):
        with self.assertRaises(FileNotFoundError):
            pf.discover_character_refs(os.path.join(AI_VIDEO_DIR, "nope"))

    def test_max_refs_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = make_images(tmp, "1.png", "2.png", "3.png", "4.png", "5.png")
            with self.assertRaises(ValueError) as ctx:
                pf.discover_character_refs(d, max_refs=4)
            self.assertIn("at most 4", str(ctx.exception))


class CliTest(unittest.TestCase):
    def _args(self, *argv):
        return main.parse_args(list(argv))

    def test_default_unset(self):
        args = self._args("--project", "w", "--prompt-dir", "prompt")
        self.assertIsNone(args.character_reference_dir)

    def test_valid_combination(self):
        args = self._args("--project", "w", "--prompt-dir", "prompt",
                          "--video-model", "minimax_h3_r2v",
                          "--character-reference-dir", "characters")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_requires_prompt_dir(self):
        args = self._args("--project", "w",
                          "--video-model", "minimax_h3_r2v",
                          "--character-reference-dir", "characters")
        self.assertIn("--prompt-dir", main.validate_cli_combination(args))

    def test_rejects_other_models(self):
        for model in ("ltx", "ltx_t2v", "minimax_h3", "wan"):
            args = self._args("--project", "w", "--prompt-dir", "prompt",
                              "--video-model", model,
                              "--character-reference-dir", "characters")
            self.assertIn("minimax_h3_r2v",
                          main.validate_cli_combination(args))

    def test_first_frame_flags_accept_r2v(self):
        for flag, value in (("--use-first-frame", None),
                            ("--image-1-dir", "image_start")):
            argv = ["--project", "w", "--prompt-dir", "prompt",
                    "--video-model", "minimax_h3_r2v", flag]
            if value is not None:
                argv.append(value)
            with self.subTest(flag=flag):
                self.assertIsNone(
                    main.validate_cli_combination(self._args(*argv)))

    def test_continuous_rejects_character_reference_dir(self):
        args = self._args("idea", "--continuous", "--image", "x.png",
                          "--character-reference-dir", "characters")
        self.assertIn("--continuous", main.validate_cli_combination(args))

    def test_missing_directory_fails_before_comfy(self):
        proj = "r2v_missing_refs_tmp"
        base = os.path.join(AI_VIDEO_DIR, "projects", proj)
        self.addCleanup(shutil.rmtree, base, True)
        os.makedirs(os.path.join(base, "prompt"), exist_ok=True)
        with open(os.path.join(base, "prompt", "1.txt"), "w",
                  encoding="utf-8") as f:
            f.write("One.")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main.main(["--project", proj,
                              "--video-model", "minimax_h3_r2v",
                              "--prompt-dir", "prompt",
                              "--character-reference-dir", "nope",
                              "--stop-after", "video"])
        self.assertEqual(code, 1)
        self.assertIn("Character reference directory not found",
                      err.getvalue())


class StageTest(unittest.TestCase):
    def test_per_clip_prompts_manifest_durations_and_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 7], "One.", "Two.")
            refs = [os.path.join(make_images(tmp, "1.png", "2.jpg"), n)
                    for n in ("1.png", "2.jpg")]
            ctx = make_ctx(tmp, make_provider(), character_refs=refs)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            reqs = []
            provider = ctx["video_provider"]
            orig = provider.generate
            provider.generate = lambda r: (reqs.append(r), orig(r))[1]
            run_stage(ctx, scenes, force_all=True)
            self.assertEqual([(r.prompt, r.duration) for r in reqs],
                             [("One.", 5), ("Two.", 7)])
            for req in reqs:
                self.assertEqual(req.character_refs, refs)
                self.assertIsNone(req.input_image)
            self.assertEqual(len(provider.client.queued), 2)

    def test_refs_shared_by_every_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 5], "One.", "Two.")
            refs = [os.path.join(make_images(tmp, "1.png"), "1.png")]
            ctx = make_ctx(tmp, make_provider(), character_refs=refs)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            names = [w["137"]["inputs"]["image"]
                     for w in ctx["video_provider"].client.queued]
            self.assertEqual(names, ["1.png", "1.png"])

    def test_initial_image_clip1_then_chain_clip2_with_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 5], "One.", "Two.")
            image_dir = make_images(tmp, "1.png")
            refs = [os.path.join(image_dir, "1.png")]
            ctx = make_ctx(tmp, make_provider(), character_refs=refs,
                           image_1_dir=image_dir, chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            queued = ctx["video_provider"].client.queued
            guides = [w["136"]["inputs"].get("ref_images.ref_image_4")
                      for w in queued]
            self.assertEqual(guides, [["1", 0], ["1", 0]])
            self.assertEqual([w["1"]["inputs"]["image"] for w in queued],
                             ["1.png", "scene_001_last.png"])
            # Character refs never overwritten by guide frames.
            self.assertEqual([w["137"]["inputs"]["image"] for w in queued],
                             ["1.png", "1.png"])

    def test_no_initial_image_keeps_clip1_refs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5], "One.")
            d = make_images(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(),
                           character_refs=[os.path.join(d, "1.png")])
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, force_all=True)
            wf = ctx["video_provider"].client.queued[-1]
            self.assertNotIn("ref_images.ref_image_4",
                             wf["136"]["inputs"])
            self.assertEqual(wf["137"]["inputs"]["image"], "1.png")

    def test_resume_skips_valid_clips_without_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 5], "One.", "Two.")
            d = make_images(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(),
                           character_refs=[os.path.join(d, "1.png")])
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, valid=True, force_all=False)
            provider = ctx["video_provider"]
            self.assertEqual(provider.client.queued, [])
            self.assertEqual(provider.client.uploads, [])

    def test_scene2_chaining_requires_previous_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 5], "One.", "Two.")
            ctx = make_ctx(tmp, make_provider(), chain=True)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            only_tmp = lambda p, *a: str(p).endswith(".tmp")
            with self.assertRaises(ValueError) as err:
                run_stage(ctx, scenes, only_tmp, only_scene=2)
            self.assertIn("scene_001.mp4 missing or invalid",
                          str(err.exception))

    def test_scene2_without_chain_uses_refs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5, 5], "One.", "Two.")
            d = make_images(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(),
                           character_refs=[os.path.join(d, "1.png")])
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            run_stage(ctx, scenes, valid=True, only_scene=2)
            wf = ctx["video_provider"].client.queued[-1]
            self.assertNotIn("ref_images.ref_image_4", wf["136"]["inputs"])
            self.assertEqual(wf["137"]["inputs"]["image"], "1.png")

    def test_logs_reflect_r2v_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5], "One.")
            d = make_images(tmp, "1.png")
            ctx = make_ctx(tmp, make_provider(),
                           character_refs=[os.path.join(d, "1.png")],
                           image_1_dir=d)
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                run_stage(ctx, scenes, force_all=True)
            text = out.getvalue()
            self.assertIn("r2v", text)
            self.assertIn("character_refs=1", text)
            self.assertIn("guide_frame=initial image", text)
            self.assertIn("prompt=1.txt", text)
            self.assertNotIn("t2v, no input image", text)

    def test_unsupported_provider_rejects_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_dir = make_prompt_dir(tmp, [5], "One.")
            from providers.video.minimax_h3 import MiniMaxH3VideoProvider
            entry = lookup(AI_VIDEO_DIR, "video", "minimax_h3")
            t2v = MiniMaxH3VideoProvider(name="minimax_h3",
                                         settings=dict(entry),
                                         client=StubClient())
            ctx = make_ctx(tmp, t2v, character_refs=["1.png"])
            ctx["video_model"] = "minimax_h3"
            scenes = pf.load_prompt_scenes(prompt_dir, 5.0)
            with self.assertRaises(ValueError) as err:
                run_stage(ctx, scenes, force_all=True)
            self.assertIn("--character-reference-dir is not supported",
                          str(err.exception))


if __name__ == "__main__":
    unittest.main()
