"""Unit tests: MiniMax H3 text-to-video backend (mocked ComfyUI, no GPU)."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
import state as st
from providers.comfy import ComfyClient, ComfyError
from providers.errors import ProviderError
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest
from providers.video.minimax_h3 import MiniMaxH3VideoProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "video", "minimax_h3")
    settings = dict(entry, **overrides)
    return MiniMaxH3VideoProvider(name="minimax_h3", settings=settings,
                                  client=ComfyClient("http://127.0.0.1:8188"))


class StubClient(ComfyClient):
    """Captures queued workflows; writes no files, touches no server."""

    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.queued = []
        self.runs = []

    def run(self, workflow, output_keys, dest_path: str) -> str:
        self.queued.append(workflow)
        self.runs.append((tuple(output_keys), dest_path))
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"fake")
        return dest_path


def workflow_inputs_contain(wf, text):
    """True if `text` appears in any input value of the workflow."""
    needle = str(text)
    for node in wf.values():
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, str) and needle in value:
                return True
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and needle in item:
                        return True
    return False


def workflow_mentions(wf, text):
    """True if `text` appears anywhere (keys excluded) in the workflow."""
    needle = str(text).lower()
    for node in wf.values():
        for key, value in (node.get("inputs") or {}).items():
            if needle in key.lower():
                return True
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, str) and needle in item.lower():
                    return True
    return False


class MinimaxCreationTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("video", AI_VIDEO_DIR, "minimax_h3",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, MiniMaxH3VideoProvider)
        self.assertEqual(provider.provider_id, "minimax_h3")
        self.assertFalse(provider.needs_input_image)

    def test_ltx_backends_unaffected(self):
        from providers.video.ltx import LtxVideoProvider
        from providers.video.ltx_t2v import LtxT2VVideoProvider
        i2v = create_provider("video", AI_VIDEO_DIR, "ltx",
                              client=ComfyClient("http://127.0.0.1:8188"))
        t2v = create_provider("video", AI_VIDEO_DIR, "ltx_t2v",
                              client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(i2v, LtxVideoProvider)
        self.assertIsInstance(t2v, LtxT2VVideoProvider)
        self.assertTrue(getattr(i2v, "needs_input_image", True))
        self.assertFalse(t2v.needs_input_image)

    def test_workflow_structure(self):
        wf = make_provider().load_workflow()
        self.assertEqual(len(wf), 23)
        classes = {n.get("class_type") for n in wf.values()}
        self.assertNotIn("LoadImage", classes)
        for required in ("MiniMaxH3ImageToVideo", "RandomNoise",
                         "PrimitiveFloat", "ResolutionSelector", "SaveVideo",
                         "CreateVideo"):
            self.assertIn(required, classes)

    def test_first_last_frame_unused(self):
        wf = make_provider().load_workflow()
        self.assertFalse(workflow_mentions(wf, "first_frame"))
        self.assertFalse(workflow_mentions(wf, "last_frame"))

    def test_scheduler_steps_linked_to_switch(self):
        # Regression (OOM investigation): the exporter once baked the
        # unconnected scheduler widget (4) as a constant, dropping inner
        # link 235. Canonical runs switch-selected steps (20, turbo off).
        wf = make_provider().load_workflow()
        sched = next(n for n in wf.values()
                     if n.get("class_type") == "BasicScheduler")
        self.assertEqual(sched["inputs"]["steps"], ["1136", 0])
        switch = wf["1136"]
        self.assertEqual(switch["inputs"]["on_false"], ["1137", 0])
        self.assertEqual(wf["1137"]["inputs"]["value"], 20)

    def test_no_ltx_frame_math(self):
        # MiniMax duration is seconds (float) through its own 17k+5
        # lattice math; the LTX seconds*24+1 rule must not appear here.
        wf = make_provider().load_workflow()
        math = next(n for n in wf.values()
                    if n.get("class_type") == "ComfyMathExpression")
        self.assertIn("17", math["inputs"]["expression"])
        self.assertNotIn("24 + 1", math["inputs"]["expression"])
        provider = make_provider()
        customized = provider.customize(
            copy.deepcopy(wf), video_prompt="P", seed=1, duration=10.0,
            aspect_label="16:9 (Widescreen)", megapixels=1.0)
        found = provider._find_nodes(customized)
        # Duration stays seconds; frame conversion lives in the graph.
        self.assertEqual(customized[found["duration"]]["inputs"]["value"], 10.0)

    def test_exporter_reproduces_shipped_workflow(self):
        # The shipped runtime graph must equal a fresh template expansion:
        # any dropped link (like scheduler steps once was) fails here.
        import export_minimax_h3_template as exporter
        self.assertEqual(exporter.build(), make_provider().load_workflow())

    def test_find_nodes(self):
        found = make_provider()._find_nodes(make_provider().load_workflow())
        self.assertNotIn("image", found)
        for key in ("generate", "seed", "duration", "resolution", "save"):
            self.assertIn(key, found)

    def test_missing_workflow_raises_provider_error(self):
        provider = make_provider(workflow_path=os.path.join(AI_VIDEO_DIR, "nope.json"))
        with self.assertRaises(ProviderError) as ctx:
            provider.load_workflow()
        self.assertIn("minimax_h3", str(ctx.exception))

    def test_image_workflow_rejected(self):
        i2v = lookup(AI_VIDEO_DIR, "video", "ltx")
        provider = make_provider(workflow_path=i2v["workflow_path"])
        with self.assertRaises(ProviderError):
            provider.load_workflow()


class MinimaxSubstitutionTest(unittest.TestCase):
    def test_only_allowed_inputs_change(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(
            template, video_prompt="P", seed=7, duration=7.0,
            aspect_label="16:9 (Widescreen)", megapixels=1.0)
        allowed = {"MiniMaxH3ImageToVideo", "RandomNoise", "PrimitiveFloat",
                   "ResolutionSelector"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        changed = [nid for nid in before if before[nid] != after[nid]]
        # prompt, seed, duration, resolution
        self.assertEqual(len(changed), 4)

    def test_template_file_unchanged_on_disk(self):
        provider = make_provider()
        with open(provider.workflow_path, "rb") as f:
            before = f.read()
        template = provider.load_workflow()
        provider.customize(template, video_prompt="P", seed=7, duration=5.0,
                           aspect_label="16:9 (Widescreen)", megapixels=1.0)
        with open(provider.workflow_path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_prompt_and_duration_injected(self):
        provider = make_provider()
        provider.client = StubClient()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="River dawn footage",
                               negative_prompt="",
                               input_image=None, width=1312, height=736,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "scene_001.mp4")
            provider.generate(req)
            wf = provider.client.queued[-1]
        self.assertTrue(workflow_inputs_contain(wf, "River dawn footage"))
        found = provider._find_nodes(wf)
        self.assertEqual(wf[found["duration"]]["inputs"]["value"], 5.0)
        # first/last frame stay unused in the submitted job too
        self.assertFalse(workflow_mentions(wf, "first_frame"))
        self.assertFalse(workflow_mentions(wf, "last_frame"))

    def test_submission_uses_video_keys(self):
        provider = make_provider()
        provider.client = StubClient()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=None, width=768, height=512,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            result = provider.generate(req)
            self.assertTrue(result.path.is_absolute() or str(result.path))
        keys, dest = provider.client.runs[-1]
        self.assertIn("video", keys)
        self.assertTrue(dest.endswith("o.mp4"))

    def test_provided_image_rejected(self):
        provider = make_provider()
        provider.client = StubClient()
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "frame.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=img, width=768, height=512,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(req)
            self.assertIn("no input image", str(ctx.exception).lower())

    def test_no_upload_attempted(self):
        class NoUploadClient(StubClient):
            def upload_image(self, image_path: str) -> str:
                raise AssertionError("T2V must not upload images")

        provider = make_provider()
        provider.client = NoUploadClient()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=None, width=768, height=512,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            provider.generate(req)
            self.assertEqual(len(provider.client.queued), 1)

    def test_nonempty_negative_rejected(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="blurry",
                           input_image=None, width=768, height=512,
                           duration=5, seed=1, output_path=Path("x.mp4"))
        with self.assertRaises(ProviderError):
            provider.generate(req)

    def test_bad_duration_rejected(self):
        provider = make_provider()
        for bad in (0, 16, 99):
            with self.subTest(bad=bad):
                req = VideoRequest(prompt="p", negative_prompt="",
                                   input_image=None, width=768, height=512,
                                   duration=bad, seed=1,
                                   output_path=Path("x.mp4"))
                with self.assertRaises(ValueError):
                    provider.generate(req)

    def test_resolution_matches_ltx_convention(self):
        from providers.video.ltx import LtxVideoProvider
        mm = make_provider()
        i2v_entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        i2v = LtxVideoProvider(name="ltx", settings=dict(i2v_entry),
                               client=ComfyClient("http://127.0.0.1:8188"))
        self.assertEqual(mm.selector_settings(1312, 736),
                         i2v.selector_settings(1312, 736))
        self.assertEqual(mm.adjust_dimensions(1300, 700),
                         i2v.adjust_dimensions(1300, 700))

    def test_comfy_error_wrapped_with_context(self):
        class BoomClient(StubClient):
            def run(self, workflow, output_keys, dest_path: str) -> str:
                raise ComfyError("connection reset")

        provider = make_provider()
        provider.client = BoomClient()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="p", negative_prompt="",
                               input_image=None, width=768, height=512,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            with self.assertRaises(ProviderError) as ctx:
                provider.generate(req)
            text = str(ctx.exception)
            self.assertIn("[provider:minimax_h3]", text)
            self.assertIn("minimax_h3_t2v.json", text)
            self.assertIn("connection reset", text)


class MinimaxStageTest(unittest.TestCase):
    class StubProvider:
        needs_input_image = False

        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            Path(str(request.output_path)).write_bytes(b"fakevideo")
            return SimpleNamespace(path=Path(str(request.output_path)),
                                   width=1312, height=736)

    def _ctx(self, tmp):
        state = st.new_state("w")
        return {"project": "w", "base_dir": tmp, "base_seed": 1,
                "state": state, "state_path": os.path.join(tmp, "s.json"),
                "img_dir": tmp, "vid_dir": tmp, "video_provider": self.StubProvider(),
                "video_model": "minimax_h3", "vid_neg": "",
                "target_w": 1312, "target_h": 736, "default_duration": 5.0}

    def _scenes(self):
        return [{"id": 1, "duration": 5, "image_prompt": "STILL IMAGE.",
                 "video_prompt": "River dawn footage.",
                 "narration": "Voice."},
                {"id": 2, "duration": 6, "image_prompt": "STILL IMAGE.",
                 "video_prompt": "Market motion.",
                 "narration": "Voice."}]

    def test_video_prompt_used_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                paths = main.run_video_stage(ctx, self._scenes(), {},
                                             force_all=True)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 2)
            self.assertEqual(prov.requests[0].prompt, "River dawn footage.")
            self.assertEqual(prov.requests[1].prompt, "Market motion.")
            for req in prov.requests:
                self.assertIsNone(req.input_image)
                self.assertNotIn("STILL IMAGE", req.prompt)
            self.assertEqual(len(paths), 2)

    def test_resume_skips_valid_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            with patch.object(st, "valid_video_file", return_value=True), \
                 patch.object(st, "valid_image_file", return_value=False):
                paths = main.run_video_stage(
                    ctx, self._scenes(), {}, force_all=False)
            self.assertEqual(len(ctx["video_provider"].requests), 0)
            self.assertEqual(len(paths), 2)

    def test_scene_selection_processes_only_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, self._scenes(), {}, only_scene=2)
            prov = ctx["video_provider"]
            self.assertEqual(len(prov.requests), 1)
            self.assertEqual(prov.requests[0].prompt, "Market motion.")


class MinimaxCliTest(unittest.TestCase):
    def _args(self, *argv):
        return main.parse_args(list(argv))

    def test_model_parses(self):
        args = self._args("--project", "w", "--resume", "--stage", "video",
                          "--video-model", "minimax_h3")
        self.assertEqual(args.video_model, "minimax_h3")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_planner_skips_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages, problems = main.plan_resume_stages(
                scenes=[{"id": 1, "duration": 5}],
                img_dir=tmp, vid_dir=tmp, aud_dir=tmp,
                final_path=os.path.join(tmp, "f.mp4"),
                default_duration=5.0, stage="video",
                video_needs_images=False)
            self.assertEqual(stages, ["video"])
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
