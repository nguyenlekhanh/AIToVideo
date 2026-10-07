"""Unit tests: LTX text-to-video backend (mocked ComfyUI, no GPU)."""
from __future__ import annotations

import copy
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
from providers.video.ltx_t2v import LtxT2VVideoProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_provider(**overrides):
    entry = lookup(AI_VIDEO_DIR, "video", "ltx_t2v")
    settings = dict(entry, **overrides)
    return LtxT2VVideoProvider(name="ltx_t2v", settings=settings,
                               client=ComfyClient("http://127.0.0.1:8188"))


class StubClient(ComfyClient):
    """Captures queued workflows; writes no files, touches no server."""

    def __init__(self):
        super().__init__("http://127.0.0.1:1")
        self.queued = []

    def run(self, workflow, output_keys, dest_path: str) -> str:
        self.queued.append(workflow)
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        Path(dest_path).write_bytes(b"fake")
        return dest_path


class T2VCreationTest(unittest.TestCase):
    def test_create_via_registry(self):
        provider = create_provider("video", AI_VIDEO_DIR, "ltx_t2v",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, LtxT2VVideoProvider)
        self.assertEqual(provider.provider_id, "ltx_t2v")
        self.assertFalse(provider.needs_input_image)

    def test_i2v_backend_unaffected(self):
        from providers.video.ltx import LtxVideoProvider
        provider = create_provider("video", AI_VIDEO_DIR, "ltx",
                                   client=ComfyClient("http://127.0.0.1:8188"))
        self.assertIsInstance(provider, LtxVideoProvider)
        self.assertTrue(getattr(provider, "needs_input_image", True))

    def test_workflow_is_image_free(self):
        wf = make_provider().load_workflow()
        self.assertEqual(len(wf), 43)  # 50 I2V nodes minus 7 removed
        classes = {n.get("class_type") for n in wf.values()}
        for removed in ("LoadImage", "LTXVPreprocess", "LTXVImgToVideoInplace",
                        "TextGenerateLTX2Prompt", "ResizeImageMaskNode"):
            self.assertNotIn(removed, classes)
        for required in ("EmptyLTXVLatentVideo", "LTXVConcatAVLatent",
                         "LTXVSeparateAVLatent", "SamplerCustomAdvanced",
                         "LTXVConditioning", "ResolutionSelector",
                         "SaveVideo"):
            self.assertIn(required, classes)

    def test_save_prefix_is_t2v(self):
        wf = make_provider().load_workflow()
        saves = [n for n in wf.values() if n.get("class_type") == "SaveVideo"]
        self.assertEqual(len(saves), 1)
        self.assertIn("t2v", saves[0]["inputs"]["filename_prefix"])
        self.assertNotIn("i2v", saves[0]["inputs"]["filename_prefix"])

    def test_find_nodes_has_no_image(self):
        found = make_provider()._find_nodes(make_provider().load_workflow())
        self.assertNotIn("image", found)
        for key in ("prompt", "negative", "seeds", "duration", "resolution"):
            self.assertIn(key, found)

    def test_i2v_workflow_rejected(self):
        i2v = lookup(AI_VIDEO_DIR, "video", "ltx")
        provider = make_provider(workflow_path=i2v["workflow_path"])
        with self.assertRaises(ProviderError):
            provider.load_workflow()

    def test_missing_workflow_raises_provider_error(self):
        provider = make_provider(workflow_path=os.path.join(AI_VIDEO_DIR, "nope.json"))
        with self.assertRaises(ProviderError) as ctx:
            provider.load_workflow()
        self.assertIn("ltx_t2v", str(ctx.exception))


class T2VSubstitutionTest(unittest.TestCase):
    def test_only_allowed_inputs_change(self):
        provider = make_provider()
        template = provider.load_workflow()
        before = copy.deepcopy(template)
        after = provider.customize(
            template, video_prompt="P", negative_prompt="N", seed=7,
            duration=7, aspect_label="16:9 (Widescreen)", megapixels=1.0)
        allowed = {"PrimitiveStringMultiline", "CLIPTextEncode", "RandomNoise",
                   "PrimitiveInt", "ResolutionSelector", "ComfySwitchNode"}
        for nid, node in before.items():
            if node != after[nid]:
                self.assertIn(node["class_type"], allowed,
                              f"Unexpected change in node {nid} ({node['class_type']})")
        changed = [nid for nid in before if before[nid] != after[nid]]
        # prompt, neg, 2x seed, duration, resolution, orphaned-switch branch
        self.assertEqual(len(changed), 6)

    def test_no_image_conditioning_in_queued_workflow(self):
        provider = make_provider()
        provider.client = StubClient()
        with tempfile.TemporaryDirectory() as tmp:
            req = VideoRequest(prompt="River dawn", negative_prompt="n",
                               input_image=None, width=1312, height=736,
                               duration=7, seed=1,
                               output_path=Path(tmp) / "scene_001.mp4")
            provider.generate(req)
            wf = provider.client.queued[-1]
        classes = [n.get("class_type") for n in wf.values()]
        self.assertNotIn("LoadImage", classes)
        texts = [n["inputs"]["value"] for n in wf.values()
                 if n.get("class_type") == "PrimitiveStringMultiline"]
        self.assertIn("River dawn", texts)
        found = provider._find_nodes(wf)
        self.assertEqual(wf[found["duration"]]["inputs"]["value"], 7)

    def test_provided_image_rejected(self):
        provider = make_provider()
        provider.client = StubClient()
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "frame.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
            req = VideoRequest(prompt="p", negative_prompt="n",
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
            req = VideoRequest(prompt="p", negative_prompt="n",
                               input_image=None, width=768, height=512,
                               duration=5, seed=1,
                               output_path=Path(tmp) / "o.mp4")
            provider.generate(req)
            self.assertEqual(len(provider.client.queued), 1)

    def test_bad_duration_rejected(self):
        provider = make_provider()
        req = VideoRequest(prompt="p", negative_prompt="n", input_image=None,
                           width=768, height=512, duration=99, seed=1,
                           output_path=Path("x.mp4"))
        with self.assertRaises(ValueError):
            provider.generate(req)

    def test_resolution_matches_i2v(self):
        from providers.video.ltx import LtxVideoProvider
        t2v = make_provider()
        i2v_entry = lookup(AI_VIDEO_DIR, "video", "ltx")
        i2v = LtxVideoProvider(name="ltx", settings=dict(i2v_entry),
                               client=ComfyClient("http://127.0.0.1:8188"))
        self.assertEqual(t2v.selector_settings(1312, 736),
                         i2v.selector_settings(1312, 736))
        self.assertEqual(t2v.adjust_dimensions(1300, 700),
                         i2v.adjust_dimensions(1300, 700))


class T2VStageTest(unittest.TestCase):
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
                "video_model": "ltx", "vid_neg": "",
                "target_w": 1312, "target_h": 736, "default_duration": 5.0}

    def _scenes(self):
        return [{"id": 1, "duration": 7, "image_prompt": "STILL IMAGE.",
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
                # No scene images on disk at all: must still work.
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

    def test_durations_reach_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._ctx(tmp)
            with patch.object(st, "valid_video_file", side_effect=lambda p, *a: str(p).endswith(".tmp")), \
                 patch.object(st, "valid_image_file", return_value=False):
                main.run_video_stage(ctx, self._scenes(), {}, force_all=True)
            self.assertEqual([r.duration for r in ctx["video_provider"].requests],
                             [7.0, 6.0])


class T2VCliTest(unittest.TestCase):
    def _args(self, *argv):
        return main.parse_args(list(argv))

    def test_default_is_i2v(self):
        self.assertEqual(self._args("hello").video_mode, "i2v")

    def test_t2v_parses(self):
        args = self._args("--project", "w", "--resume", "--stage", "video",
                          "--video-model", "ltx", "--video-mode", "t2v")
        self.assertEqual(args.video_mode, "t2v")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_t2v_rejects_non_ltx(self):
        args = self._args("--project", "w", "--resume", "--stage", "video",
                          "--video-model", "wan", "--video-mode", "t2v")
        err = main.validate_cli_combination(args)
        self.assertIsNotNone(err)
        self.assertIn("ltx", err)

    def test_planner_skips_images_for_t2v(self):
        with tempfile.TemporaryDirectory() as tmp:
            stages, problems = main.plan_resume_stages(
                scenes=[{"id": 1, "duration": 7}],
                img_dir=tmp, vid_dir=tmp, aud_dir=tmp,
                final_path=os.path.join(tmp, "f.mp4"),
                default_duration=5.0, stage="video",
                video_needs_images=False)
            self.assertEqual(stages, ["video"])
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
