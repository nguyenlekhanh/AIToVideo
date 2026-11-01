"""Focused unit tests for the opt-in product-URL auto-ad pipeline.

Network, Ollama, ComfyUI, FFmpeg and TTS are all mocked here. The real
acceptance run is deliberately separate.
"""
from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import ad_planner as ap
import ffmpeg as ff
import main
import ollama as ol
import product_research as pr


PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
               ("93.184.216.34", 443))]


def png_bytes(size=(400, 400), color=(40, 120, 200)):
    image = Image.new("RGB", size, color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def write_png(path, size=(400, 400), color=(40, 120, 200)):
    with open(path, "wb") as f:
        f.write(png_bytes(size=size, color=color))
    return path


def make_record(tmp):
    refs = []
    for index in range(1, 4):
        path = write_png(os.path.join(tmp, f"product_{index:02d}.png"),
                         color=(index * 40, 120, 200))
        refs.append({
            "id": f"img_{index}",
            "path": path,
            "source_url": f"https://cdn.example.com/{index}.webp",
            "sha256": f"hash{index}",
            "verification": {"matches": True, "confidence": 0.9},
        })
    return {
        "version": 1,
        "source_url": "https://shop.example.com/p/1",
        "canonical_url": "https://shop.example.com/p/1",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "retrieval_status": "complete",
        "source_type": "product_page",
        "title": "Test Widget - Blue",
        "display_name": "Test Widget",
        "brand": "Acme",
        "description": "A test product.",
        "category": "Test",
        "price": {"amount": 9.99, "currency": "USD", "display": "$9.99"},
        "rating": None,
        "review_count": None,
        "sold_count": None,
        "features": [
            {"id": "f1", "label": "Adjustable",
             "text": "Adjustable sizes for a precise fit",
             "source_url": "https://shop.example.com/p/1",
             "source_type": "product_page", "confidence": "high"},
            {"id": "f2", "label": "Safety",
             "text": "Safety guard keeps trimming clean and controlled",
             "source_url": "https://shop.example.com/p/1",
             "source_type": "product_page", "confidence": "high"},
            {"id": "f3", "label": "Title",
             "text": "Safe, simple cat grooming",
             "source_url": "https://shop.example.com/p/1",
             "source_type": "product_page", "confidence": "high"},
        ],
        "image_candidates": [],
        "selected_images": refs,
        "sources": [{"id": "src_1", "url": "https://shop.example.com/p/1",
                     "title": "Test Widget", "source_type": "product_page",
                     "confidence": "high"}],
        "facts": [
            {"id": "fact_1", "field": "title", "value": "Test Widget",
             "source_id": "src_1", "confidence": "high"},
        ],
        "fingerprint": "product-fingerprint",
    }


def base_plan():
    return {
        "ad_title": "Test Widget for Cat Grooming",
        "audience": "Cat owners",
        "core_message": "A practical grooming tool with verified details.",
        "clips": [
            {"id": 1, "role": "hook",
             "image_prompt": "A close-up of the Test Widget on a table.",
             "video_prompt": "A hand picks up the Test Widget and turns it.",
             "spoken_text": "Trim your cat's claws at home with four adjustable sizes for a precise fit.",
             "on_screen_text": None,
             "continuity": "Introduce the product.",
             "feature_ids": ["f1"],
             "product_reference_ids": ["img_1"]},
            {"id": 2, "role": "feature",
             "image_prompt": "A close-up of the safety guard.",
             "video_prompt": "The safety guard is shown while trimming.",
             "spoken_text": "The safety guard helps keep trimming clean and controlled during every grooming session.",
             "on_screen_text": None,
             "continuity": "Show the safety guard.",
             "feature_ids": ["f2"],
             "product_reference_ids": ["img_2"]},
            {"id": 3, "role": "cta",
             "image_prompt": "The Test Widget on a clean background.",
             "video_prompt": "The product is shown from the front.",
             "spoken_text": "Check the product page for safe, simple cat grooming that fits your routine.",
             "on_screen_text": None,
             "continuity": "End with the product.",
             "feature_ids": ["f3"],
             "product_reference_ids": ["img_3"]},
        ],
        "claims_used": [
            {"text": "Adjustable sizes", "feature_ids": ["f1"]},
            {"text": "Safety guard keeps trimming clean",
             "feature_ids": ["f2"]},
        ],
        "claims_avoided": ["No invented medical claims."],
        "cta": "Check the product page for safe, simple cat grooming.",
        "audio_plan": {"mode": "per_clip_voiceover"},
    }


class UrlSafetyTest(unittest.TestCase):
    def test_public_https_is_normalized(self):
        with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
            self.assertEqual(
                pr.validate_product_url(
                    "HTTPS://Example.COM/a?b=1#fragment"),
                "https://example.com/a?b=1")

    def test_rejects_unsupported_scheme(self):
        with self.assertRaises(pr.UnsafeUrlError):
            pr.validate_product_url("file:///C:/secret.txt")

    def test_rejects_credentials(self):
        with self.assertRaises(pr.UnsafeUrlError):
            pr.validate_product_url("https://user:pass@example.com/p")

    def test_rejects_private_destination(self):
        with patch.object(pr.socket, "getaddrinfo",
                          return_value=[(socket.AF_INET, socket.SOCK_STREAM,
                                         6, "", ("127.0.0.1", 443))]):
            with self.assertRaises(pr.UnsafeUrlError):
                pr.validate_product_url("https://example.com/p")

    def test_rejects_localhost_name_without_dns(self):
        with self.assertRaises(pr.UnsafeUrlError):
            pr.validate_product_url("http://localhost/p")

    def test_redirect_to_private_host_is_rejected(self):
        class FakeResponse:
            status_code = 302
            headers = {"Location": "http://127.0.0.1/secret"}

            def close(self):
                pass

        class FakeSession:
            def get(self, *args, **kwargs):
                return FakeResponse()

            def close(self):
                pass

        def fake_dns(host, port, *args, **kwargs):
            if host == "127.0.0.1":
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, "",
                         ("127.0.0.1", port))]
            return PUBLIC_DNS

        with patch.object(pr.requests, "Session", return_value=FakeSession()), \
             patch.object(pr.socket, "getaddrinfo", side_effect=fake_dns):
            with self.assertRaises(pr.UnsafeUrlError):
                pr._request_bytes("https://example.com/p", timeout=5,
                                  max_bytes=1000, accept="text/html")

    def test_derive_project_name_is_safe(self):
        self.assertEqual(
            pr.derive_project_name(
                "https://shop.example.com/p/cat-clipper/12345"),
            "ad_cat_clipper")


class ProductParsingTest(unittest.TestCase):
    def _tiktok_html(self):
        product = {
            "title": "Test Widget - Blue",
            "images": [{
                "width": 800, "height": 800,
                "url_list": ["https://cdn.example.com/a.webp"],
            }],
            "price": {"real_price": "$12.97", "currency": "USD"},
            "specifications": [{"name": "Material", "value": "ABS"}],
            "desc_blocks": [{
                "type": "text",
                "text": "Adjustable Grip: The handle adjusts for a precise fit.",
            }],
            "seller": {"name": "Acme"},
            "review": {"product_rating": 4.8, "review_count": 10},
            "sold_count": 100,
        }
        router = {
            "loaderData": {
                "page": {
                    "page_config": {
                        "components_map": [{
                            "component_data": {"product_info": product}
                        }]
                    }
                }
            }
        }
        return (
            '<script id="__MODERN_ROUTER_DATA__" type="application/json">'
            + json.dumps(router) + "</script>")

    def test_tiktok_router_data_extraction(self):
        with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
            record = pr.parse_product_html(
                self._tiktok_html(), "https://shop.example.com/p/1")
        self.assertEqual(record["title"], "Test Widget - Blue")
        self.assertEqual(record["display_name"], "Test Widget")
        self.assertEqual(record["brand"], "Acme")
        self.assertEqual(record["price"]["amount"], 12.97)
        self.assertEqual(record["price"]["currency"], "USD")
        self.assertEqual(record["rating"], 4.8)
        self.assertEqual(record["review_count"], 10)
        self.assertEqual(record["sold_count"], 100)
        self.assertEqual(record["source_type"], "tiktok_shop_product")
        self.assertTrue(any("Material: ABS" == f["text"]
                            for f in record["features"]))
        self.assertEqual(len(record["image_candidates"]), 1)
        self.assertEqual(record["image_candidates"][0]["width"], 800)
        self.assertEqual(record["sources"][0]["id"], "src_1")
        self.assertTrue(record["facts"])

    def test_json_ld_extraction(self):
        html = (
            '<script type="application/ld+json">'
            + json.dumps({
                "@context": "https://schema.org",
                "@type": "Product",
                "name": "Widget",
                "brand": {"name": "Acme"},
                "description": "A useful widget for testing.",
                "image": ["https://cdn.example.com/a.jpg"],
                "offers": {"price": "9.99", "priceCurrency": "USD"},
            })
            + "</script>")
        with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
            record = pr.parse_product_html(
                html, "https://shop.example.com/p/2")
        self.assertEqual(record["title"], "Widget")
        self.assertEqual(record["brand"], "Acme")
        self.assertEqual(record["price"]["amount"], 9.99)
        self.assertEqual(record["source_type"], "product_page")
        self.assertEqual(len(record["image_candidates"]), 1)

    def test_meta_fallback(self):
        html = (
            '<meta property="og:title" content="Meta Widget">'
            '<meta property="og:description" content="A meta description.">'
            '<meta property="og:image" content="https://cdn.example.com/m.png">'
            '<meta property="product:price:amount" content="4.50">'
            '<meta property="product:price:currency" content="USD">')
        with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
            record = pr.parse_product_html(
                html, "https://shop.example.com/p/3")
        self.assertEqual(record["title"], "Meta Widget")
        self.assertEqual(record["price"]["amount"], 4.50)
        self.assertEqual(record["retrieval_status"], "partial")

    def test_missing_title_fails(self):
        with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
            with self.assertRaises(pr.ProductResearchError):
                pr.parse_product_html("<html></html>",
                                      "https://shop.example.com/p/4")


class ImageSelectionTest(unittest.TestCase):
    def _record(self):
        return {
            "title": "Test Widget",
            "brand": "Acme",
            "canonical_url": "https://shop.example.com/p/1",
            "image_candidates": [
                {"id": "img_1", "url": "https://cdn.example.com/1.png"},
                {"id": "img_2", "url": "https://cdn.example.com/2.png"},
            ],
            "selected_images": [],
        }

    def test_download_and_vision_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record()
            with patch.object(pr, "_download_image",
                              return_value=png_bytes()), \
                 patch.object(pr.ol, "check_model_available"), \
                 patch.object(pr, "_vision_verify",
                              return_value={"matches": True,
                                            "confidence": 0.95,
                                            "reason": "match"}):
                result = pr.download_and_verify_images(
                    record, project_dir=tmp, vision_model="qwen3-vl:8b",
                    require_vision=True)
            self.assertEqual(len(result["selected_images"]), 2)
            self.assertTrue(all(os.path.isfile(i["path"])
                                for i in result["selected_images"]))
            self.assertEqual(result["selected_images"][0]["verification"]
                             ["confidence"], 0.95)
            self.assertTrue(os.path.isfile(os.path.join(
                tmp, "source", "images", "product_01.png")))

    def test_interrupted_download_can_adopt_identical_existing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record()
            record["image_candidates"] = record["image_candidates"][:1]
            os.makedirs(os.path.join(tmp, "product_reference"))
            normalised, _fmt, _w, _h = pr._normalise_image(png_bytes())
            with open(os.path.join(tmp, "product_reference", "product_01.png"),
                      "wb") as f:
                f.write(normalised)
            with patch.object(pr, "_download_image",
                              return_value=png_bytes()), \
                 patch.object(pr.ol, "check_model_available"):
                result = pr.download_and_verify_images(
                    record, project_dir=tmp, vision_model=None,
                    require_vision=False)
            self.assertEqual(len(result["selected_images"]), 1)
            self.assertEqual(result["selected_images"][0]["path"],
                             os.path.join(tmp, "product_reference",
                                          "product_01.png"))

    def test_vision_rejection_leaves_no_selected_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self._record()
            with patch.object(pr, "_download_image",
                              return_value=png_bytes()), \
                 patch.object(pr.ol, "check_model_available"), \
                 patch.object(pr, "_vision_verify",
                              return_value={"matches": False,
                                            "confidence": 0.9,
                                            "reason": "different product"}):
                with self.assertRaises(pr.ProductResearchError):
                    pr.download_and_verify_images(
                        record, project_dir=tmp, vision_model="qwen3-vl:8b",
                        require_vision=True)
            self.assertFalse(os.path.exists(os.path.join(
                tmp, "product_reference", "product_01.png")))

    def test_research_reuses_same_url_record_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            record["source_url"] = "https://shop.example.com/p/1"
            os.makedirs(os.path.join(tmp, "source"), exist_ok=True)
            with open(os.path.join(tmp, "source", "product.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f)
            with patch.object(pr, "_request_bytes",
                              side_effect=AssertionError("network used")), \
                 patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
                reused = pr.research_product(
                    "https://shop.example.com/p/1", project_dir=tmp,
                    require_vision=False)
            self.assertEqual(reused["fingerprint"], "product-fingerprint")

    def test_changed_url_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            record["source_url"] = "https://shop.example.com/p/old"
            os.makedirs(os.path.join(tmp, "source"), exist_ok=True)
            with open(os.path.join(tmp, "source", "product.json"), "w",
                      encoding="utf-8") as f:
                json.dump(record, f)
            with patch.object(pr.socket, "getaddrinfo", return_value=PUBLIC_DNS):
                with self.assertRaises(pr.ProductResearchError):
                    pr.research_product(
                        "https://shop.example.com/p/new", project_dir=tmp,
                        require_vision=False)


class PlanValidationTest(unittest.TestCase):
    def test_duration_lattice_and_clip_count(self):
        self.assertAlmostEqual(ap.quantized_duration_seconds(5), 124 / 24)
        self.assertEqual(ap.choose_clip_count(30), 6)
        self.assertEqual(ap.choose_clip_count(15), 3)
        self.assertAlmostEqual(
            ap.quantized_duration_seconds(5) * 6, 31.0)

    def test_valid_plan_is_grounded_and_uses_all_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = ap.validate_plan(
                base_plan(), record, target_duration=15,
                style="ugc_product_showcase", language="en",
                clip_count=3, clip_duration=5.0, model="qwen3:8b")
        self.assertEqual(plan["clip_count"], 3)
        self.assertEqual(plan["actual_duration_seconds"], 15.5)
        for clip in plan["clips"]:
            self.assertEqual(clip["product_reference_ids"],
                             ["img_1", "img_2", "img_3"])

    def test_image_prompt_id_is_replaced_with_a_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = base_plan()
            plan["clips"][0]["image_prompt"] = "img_2"
            result = ap.validate_plan(
                plan, record, target_duration=15,
                style="ugc_product_showcase", language="en",
                clip_count=3, clip_duration=5.0, model="qwen3:8b")
        self.assertTrue(result["clips"][0]["image_prompt"].startswith(
            "Opening frame:"))
        self.assertIn("Test Widget", result["clips"][0]["image_prompt"])

    def test_unknown_feature_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = base_plan()
            plan["clips"][0]["feature_ids"] = ["f99"]
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    plan, record, target_duration=15,
                    style="ugc_product_showcase", language="en",
                    clip_count=3, clip_duration=5.0, model="qwen3:8b")

    def test_unsupported_risky_claim_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = base_plan()
            plan["clips"][0]["feature_ids"] = []
            plan["clips"][0]["spoken_text"] = (
                "This is the best cat trimmer ever made for every pet.")
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    plan, record, target_duration=15,
                    style="ugc_product_showcase", language="en",
                    clip_count=3, clip_duration=5.0, model="qwen3:8b")

    def test_later_clip_cannot_cut_to_new_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = base_plan()
            plan["clips"][1]["video_prompt"] = (
                "Cut to a new scene of a cat walking on a carpet.")
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    plan, record, target_duration=15,
                    style="ugc_product_showcase", language="en",
                    clip_count=3, clip_duration=5.0, model="qwen3:8b")

    def test_supporting_feature_is_added_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = base_plan()
            plan["clips"][1]["feature_ids"] = ["f1"]
            validated = ap.validate_plan(
                plan, record, target_duration=15,
                style="ugc_product_showcase", language="en",
                clip_count=3, clip_duration=5.0, model="qwen3:8b")
        self.assertIn("f2", validated["clips"][1]["feature_ids"])

    def test_write_prompt_v2_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = ap.validate_plan(
                base_plan(), record, target_duration=15,
                style="ugc_product_showcase", language="en",
                clip_count=3, clip_duration=5.0, model="qwen3:8b")
            result = ap.write_prompt_v2(tmp, record, plan, overwrite=True)
            prompt_dir = result["prompt_dir"]
            first = open(os.path.join(prompt_dir, "1.txt"),
                         encoding="utf-8").read()
            self.assertIn("<Picture 1>, <Picture 2>, <Picture 3>", first)
            self.assertIn("<Picture 4>", first)
            with open(os.path.join(prompt_dir, "manifest.json"),
                      encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest, {"clips": {
                "1": {"duration": 5.0},
                "2": {"duration": 5.0},
                "3": {"duration": 5.0},
            }})
            self.assertTrue(os.path.isfile(os.path.join(
                prompt_dir, "ad_plan.json")))
            self.assertEqual(ap.load_ad_plan(tmp)["clip_count"], 3)

            with open(os.path.join(prompt_dir, "1.txt"), "a",
                      encoding="utf-8") as f:
                f.write("\nmanual edit\n")
            with self.assertRaises(ap.AdPlanError):
                ap.write_prompt_v2(tmp, record, plan, overwrite=False)

            with open(os.path.join(prompt_dir, "9.txt"), "w",
                      encoding="utf-8") as f:
                f.write("stale")
            ap.write_prompt_v2(tmp, record, plan, overwrite=True)
            self.assertFalse(os.path.exists(os.path.join(prompt_dir, "9.txt")))

    def test_existing_unmanaged_prompt_dir_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = ap.validate_plan(
                base_plan(), record, target_duration=15,
                style="ugc_product_showcase", language="en",
                clip_count=3, clip_duration=5.0, model="qwen3:8b")
            os.makedirs(os.path.join(tmp, "prompt"))
            with open(os.path.join(tmp, "prompt", "1.txt"), "w",
                      encoding="utf-8") as f:
                f.write("user prompt")
            with self.assertRaises(ap.AdPlanError):
                ap.write_prompt_v2(tmp, record, plan, overwrite=False)


class AudioFitTest(unittest.TestCase):
    def test_short_narration_is_padded_to_clip_duration(self):
        with patch.object(ff, "probe_duration", return_value=3.5), \
             patch.object(ff, "run") as run, \
             patch.object(ff.os, "replace") as replace:
            ff.fit_audio_duration("in.mp3", 124 / 24, out_path="out.mp3")
        cmd = run.call_args[0][0]
        self.assertIn("apad=whole_dur=5.166667", cmd)
        self.assertIn("-t", cmd)
        self.assertIn("5.166667", cmd)
        self.assertNotIn("atempo", ",".join(cmd))
        replace.assert_called_once()

    def test_long_narration_is_gently_sped_up(self):
        with patch.object(ff, "probe_duration", return_value=6.2), \
             patch.object(ff, "run") as run, \
             patch.object(ff.os, "replace"):
            ff.fit_audio_duration("in.mp3", 124 / 24, out_path="out.mp3")
        cmd = run.call_args[0][0]
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("atempo=", af)
        self.assertIn("apad=whole_dur=5.166667", af)


class ResumeGuardTest(unittest.TestCase):
    RENDER = {"width": 480, "height": 864, "aspect": "9:16",
              "resolution": "480", "video_model": "minimax_h3_r2v"}
    AUDIO = {"model": "piper_en", "voice": "en_US-amy-medium"}

    @staticmethod
    def _state(render=None, audio=None):
        state = {"stages": {"video": {}, "audio": {}}}
        if render is not None:
            state["stages"]["video"]["render"] = render
        if audio is not None:
            state["stages"]["audio"]["render"] = audio
        return state

    def _problem(self, state, stages=("finalize",), probe=None, clip="c.mp4"):
        return main.auto_ad_resume_problem(
            state, list(stages), dict(self.RENDER), dict(self.AUDIO),
            first_clip_path=clip,
            probe_size=probe or (lambda _p: (480, 864)))

    def test_matching_state_is_reused(self):
        state = self._state(self.RENDER, self.AUDIO)
        self.assertIsNone(self._problem(state))

    def test_changed_resolution_is_rejected(self):
        state = self._state(dict(self.RENDER, width=736, height=1312))
        self.assertIn("different", self._problem(state))

    def test_missing_metadata_uses_first_clip_size(self):
        problem = self._problem(self._state(), probe=lambda _p: (736, 1312))
        self.assertIn("736x1312", problem)

    def test_missing_metadata_accepts_matching_clip_size(self):
        self.assertIsNone(self._problem(self._state()))

    def test_changed_voice_is_rejected(self):
        state = self._state(audio={"model": "edge_tts",
                                   "voice": "en-US-AriaNeural"})
        self.assertIn("voice", self._problem(state))

    def test_video_only_stage_ignores_audio_mismatch(self):
        state = self._state(audio={"model": "edge_tts", "voice": "x"})
        self.assertIsNone(self._problem(state, stages=("video",)))

    def test_metadata_survives_state_roundtrip(self):
        """state.load_state must keep the render metadata it is given."""
        import state as st
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            state = self._state(self.RENDER, self.AUDIO)
            state["project"] = "p"
            st.save_state(path, state)
            loaded = st.load_state(path, "p")
        self.assertEqual(loaded["stages"]["video"]["render"], self.RENDER)
        self.assertEqual(loaded["stages"]["audio"]["render"], self.AUDIO)


class ChatJsonTest(unittest.TestCase):
    def test_plain_json_content(self):
        with patch.object(ol, "_post",
                          return_value={"message": {"content": '{"a": 1}'}}):
            self.assertEqual(ol.chat_json("s", "u", model="m"), {"a": 1})

    def test_truncated_response_raises(self):
        with patch.object(ol, "_post",
                          return_value={"done_reason": "length",
                                        "message": {"content": '{"a": 1}'}}):
            with self.assertRaises(RuntimeError) as ctx:
                ol.chat_json("s", "u", model="m")
        self.assertIn("truncated", str(ctx.exception))

    def test_schema_is_forwarded_as_structured_output(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}},
                  "required": ["a"]}
        with patch.object(ol, "_post",
                          return_value={"message": {"content": '{"a": 1}'}}) as post:
            ol.chat_json("s", "u", model="m", schema=schema)
        payload = post.call_args[0][1]
        self.assertEqual(payload["format"], schema)
        with patch.object(ol, "_post",
                          return_value={"message": {"content": '{"a": 1}'}}) as post:
            ol.chat_json("s", "u", model="m")
        self.assertEqual(post.call_args[0][1]["format"], "json")

    def test_thinking_fallback_is_json_only(self):
        with patch.object(ol, "_post",
                          return_value={"message": {"content": "",
                                                    "thinking": '{"b": 2}'}}):
            self.assertEqual(ol.chat_json("s", "u", model="m"), {"b": 2})
        with patch.object(ol, "_post",
                          return_value={"message": {"content": "",
                                                    "thinking": "prose only"}}):
            with self.assertRaises(RuntimeError):
                ol.chat_json("s", "u", model="m")


class AutoAdCliTest(unittest.TestCase):
    URL = "https://shop.example.com/p/1"

    def _args(self, *extra):
        return main.parse_args([
            "--auto-ad", "--product-url", self.URL,
            "--video-model", "minimax_h3_r2v", *extra])

    def test_valid_auto_ad_combination(self):
        args = self._args("--ad-duration", "30", "--aspect", "9:16",
                          "--resolution", "480", "--stop-after", "video")
        self.assertIsNone(main.validate_cli_combination(args))
        self.assertTrue(args.auto_ad)
        self.assertEqual(args.ad_duration, 30.0)

    def test_product_url_required(self):
        args = main.parse_args(["--auto-ad", "--video-model",
                                "minimax_h3_r2v"])
        self.assertIn("--product-url", main.validate_cli_combination(args))

    def test_wrong_video_model_rejected(self):
        args = main.parse_args(["--auto-ad", "--product-url", self.URL,
                                "--video-model", "ltx"])
        self.assertIn("minimax_h3_r2v", main.validate_cli_combination(args))

    def test_storyboard_stop_after_rejected(self):
        args = self._args("--stop-after", "storyboard")
        self.assertIn("--stop-after", main.validate_cli_combination(args))

    def test_overwrite_outside_auto_ad_rejected(self):
        args = main.parse_args(["a prompt", "--overwrite"])
        self.assertIn("--overwrite", main.validate_cli_combination(args))

    def test_character_reference_dir_rejected(self):
        args = self._args("--character-reference-dir", "chars")
        self.assertIn("character-reference-dir",
                      main.validate_cli_combination(args))

    def test_character_reference_rejected(self):
        args = self._args("--character-reference", "face.png")
        self.assertIn("character-reference",
                      main.validate_cli_combination(args))

    def test_manual_initial_image_dir_rejected(self):
        args = self._args("--image-1-dir", "starts")
        self.assertIn("image-1-dir", main.validate_cli_combination(args))


class StyleEnergyValidationTest(unittest.TestCase):
    URL = "https://shop.example.com/p/1"

    def _args(self, *extra):
        return main.parse_args([
            "--auto-ad", "--product-url", self.URL,
            "--video-model", "minimax_h3_r2v", *extra])

    def test_cli_defaults(self):
        args = self._args()
        self.assertEqual(args.ad_style, "hybrid_ugc_cinematic")
        self.assertEqual(args.ad_energy, "viral")
        self.assertIsNone(main.validate_cli_combination(args))

    def test_every_style_energy_pair_is_valid(self):
        self.assertEqual(len(ap.AD_STYLES), 4)
        self.assertEqual(len(ap.AD_ENERGIES), 3)
        for style in ap.AD_STYLES:
            for energy in ap.AD_ENERGIES:
                with self.subTest(style=style, energy=energy):
                    self.assertEqual(
                        ap.validate_style_energy(style, energy),
                        (style, energy))
                    args = self._args("--ad-style", style,
                                      "--ad-energy", energy)
                    self.assertIsNone(main.validate_cli_combination(args))

    def test_invalid_style_rejected_by_cli(self):
        with self.assertRaises(SystemExit):
            self._args("--ad-style", "corporate")

    def test_invalid_energy_rejected_by_cli(self):
        with self.assertRaises(SystemExit):
            self._args("--ad-energy", "extreme")

    def test_combination_validator_rejects_bad_values(self):
        args = self._args()
        args.ad_style = "corporate"
        self.assertIn("ad-style", main.validate_cli_combination(args))
        args.ad_style = "hybrid_ugc_cinematic"
        args.ad_energy = "extreme"
        self.assertIn("ad-energy", main.validate_cli_combination(args))

    def test_invalid_values_fail_before_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            analysis = {
                "audience": "Cat owners", "problem": "Messy trimming",
                "desired_outcome": "Clean trimming",
                "verified_benefits": [{"text": "Adjustable sizes",
                                       "feature_ids": ["f1"]}],
                "objections": [], "recommended_angle": "Show the sizes",
                "claims_to_avoid": [],
            }
            with patch.object(ap.ol, "chat_json") as chat:
                with self.assertRaises(ap.AdPlanError):
                    ap.plan_ad(record, analysis, model="m",
                               style="corporate", energy="viral")
                chat.assert_not_called()
            with patch.object(ap.ol, "chat_json") as chat:
                with self.assertRaises(ap.AdPlanError):
                    ap.analyze_product(record, model="m",
                                       style="viral_ugc", energy="extreme")
                chat.assert_not_called()

    def test_creative_brief_reaches_both_ollama_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            analysis = {
                "audience": "Cat owners", "problem": "Messy trimming",
                "desired_outcome": "Clean trimming",
                "verified_benefits": [{"text": "Adjustable sizes",
                                       "feature_ids": ["f1"]}],
                "objections": [], "recommended_angle": "Show the sizes",
                "claims_to_avoid": [],
            }
            with patch.object(ap.ol, "chat_json",
                              side_effect=RuntimeError("stop")) as chat:
                with self.assertRaises(ap.AdPlanError):
                    ap.analyze_product(record, model="m",
                                       style="premium_commercial",
                                       energy="premium")
                analysis_user = chat.call_args[0][1]
                self.assertIn("AD STYLE: premium_commercial", analysis_user)
                self.assertIn("AD ENERGY: premium", analysis_user)
            with patch.object(ap.ol, "chat_json",
                              side_effect=RuntimeError("stop")) as chat:
                with self.assertRaises(ap.AdPlanError):
                    ap.plan_ad(record, analysis, model="m",
                               style="storytelling", energy="natural")
                plan_user = chat.call_args[0][1]
                self.assertIn("AD STYLE: storytelling", plan_user)
                self.assertIn("AD ENERGY: natural", plan_user)
                self.assertIn("Required shot mix:", plan_user)


def styled_plan(style, energy, shots=None, roles=None):
    plan = base_plan()
    if roles:
        for clip, role in zip(plan["clips"], roles):
            clip["role"] = role
    if shots:
        for clip, shot in zip(plan["clips"], shots):
            clip["shot_type"] = shot
    for clip in plan["clips"]:
        clip["camera"] = "Slow dolly in, product centred, shallow depth of field."
    plan["style"] = style
    plan["ad_style"] = style
    plan["ad_energy"] = energy
    plan["style_rationale"] = "This plan expresses the selected style."
    plan["energy_rationale"] = "This plan expresses the selected energy."
    plan["clip_count"] = len(plan["clips"])
    plan["clip_duration_seconds"] = 5.0
    return plan


class PlanStyleTest(unittest.TestCase):
    def test_new_style_plan_serialises_style_energy_and_rationales(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = ap.validate_plan(
                styled_plan("hybrid_ugc_cinematic", "viral",
                            shots=["human_ugc", "product_showcase", "hybrid"]),
                record, target_duration=15, style="hybrid_ugc_cinematic",
                energy="viral", language="en", clip_count=3,
                clip_duration=5.0, model="qwen3:8b")
        self.assertEqual(plan["ad_style"], "hybrid_ugc_cinematic")
        self.assertEqual(plan["ad_energy"], "viral")
        self.assertEqual(plan["style"], "hybrid_ugc_cinematic")
        self.assertTrue(plan["style_rationale"])
        self.assertTrue(plan["energy_rationale"])
        self.assertEqual([c["shot_type"] for c in plan["clips"]],
                         ["human_ugc", "product_showcase", "hybrid"])
        self.assertTrue(all(c["camera"] for c in plan["clips"]))

    def test_hybrid_rejects_all_talking_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            with self.assertRaises(ap.AdPlanError) as ctx:
                ap.validate_plan(
                    styled_plan("hybrid_ugc_cinematic", "viral",
                                shots=["human_ugc"] * 3),
                    record, target_duration=15,
                    style="hybrid_ugc_cinematic", energy="viral",
                    language="en", clip_count=3, clip_duration=5.0,
                    model="qwen3:8b")
        self.assertIn("human UGC", str(ctx.exception))

    def test_hybrid_rejects_all_product_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    styled_plan("hybrid_ugc_cinematic", "viral",
                                shots=["product_showcase"] * 3),
                    record, target_duration=15,
                    style="hybrid_ugc_cinematic", energy="viral",
                    language="en", clip_count=3, clip_duration=5.0,
                    model="qwen3:8b")

    def test_viral_requires_a_creator_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    styled_plan("viral_ugc", "viral",
                                shots=["product_showcase"] * 3),
                    record, target_duration=15, style="viral_ugc",
                    energy="viral", language="en", clip_count=3,
                    clip_duration=5.0, model="qwen3:8b")

    def test_premium_commercial_stays_product_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    styled_plan("premium_commercial", "premium",
                                shots=["human_ugc"] * 3),
                    record, target_duration=15,
                    style="premium_commercial", energy="premium",
                    language="en", clip_count=3, clip_duration=5.0,
                    model="qwen3:8b")

    def test_storytelling_requires_problem_and_cta(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    styled_plan("storytelling", "natural",
                                roles=["feature", "demo", "benefit"]),
                    record, target_duration=15, style="storytelling",
                    energy="natural", language="en", clip_count=3,
                    clip_duration=5.0, model="qwen3:8b")

    def test_missing_shot_type_rejected_for_new_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = styled_plan("viral_ugc", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            del plan["clips"][0]["shot_type"]
            with self.assertRaises(ap.AdPlanError):
                ap.validate_plan(
                    plan, record, target_duration=15, style="viral_ugc",
                    energy="viral", language="en", clip_count=3,
                    clip_duration=5.0, model="qwen3:8b")

    def test_product_to_human_jump_rejected_for_new_plans(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = styled_plan("hybrid_ugc_cinematic", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "human_ugc"])
            with self.assertRaises(ap.AdPlanError) as ctx:
                ap.validate_plan(
                    plan, record, target_duration=15,
                    style="hybrid_ugc_cinematic", energy="viral",
                    language="en", clip_count=3, clip_duration=5.0,
                    model="qwen3:8b", strict_continuity=True)
            self.assertIn("bridging", str(ctx.exception))
            # Existing rendered plans must still load for resume.
            ap.validate_plan(
                plan, record, target_duration=15,
                style="hybrid_ugc_cinematic", energy="viral",
                language="en", clip_count=3, clip_duration=5.0,
                model="qwen3:8b")

    def test_hybrid_bridge_after_product_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = styled_plan("hybrid_ugc_cinematic", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            result = ap.validate_plan(
                plan, record, target_duration=15,
                style="hybrid_ugc_cinematic", energy="viral",
                language="en", clip_count=3, clip_duration=5.0,
                model="qwen3:8b", strict_continuity=True)
        self.assertEqual([c["shot_type"] for c in result["clips"]],
                         ["human_ugc", "product_showcase", "hybrid"])

    def test_retry_feedback_lists_valid_roles_and_shot_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            bad = styled_plan("hybrid_ugc_cinematic", "viral",
                              shots=["human_ugc", "product_showcase",
                                     "hybrid"])
            bad["clips"][0]["role"] = "desired_outcome"
            good = styled_plan("hybrid_ugc_cinematic", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            with patch.object(ap.ol, "chat_json",
                              side_effect=[bad, good]) as chat:
                plan = ap.plan_ad(record, {}, model="m",
                                  target_duration=15,
                                  style="hybrid_ugc_cinematic",
                                  energy="viral", clip_count=3,
                                  clip_duration=5.0)
        self.assertEqual(plan["ad_style"], "hybrid_ugc_cinematic")
        second_prompt = chat.call_args_list[1][0][1]
        self.assertIn("role MUST be one of", second_prompt)
        self.assertIn("desired_outcome", second_prompt)
        self.assertIn("shot_type MUST be one of", second_prompt)
        self.assertIn("Required shot mix:", second_prompt)

    def test_prompt_text_differs_across_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            refs = record["selected_images"]
            prompts = {}
            for style in ap.AD_STYLES:
                plan = styled_plan(style, "viral",
                                   shots=["human_ugc", "product_showcase",
                                          "hybrid"])
                prompts[style] = ap._build_clip_prompt(
                    plan, plan["clips"][0], record, refs, 3)
            self.assertEqual(len(set(prompts.values())), 4)
            self.assertIn("HYBRID UGC CINEMATIC", prompts["hybrid_ugc_cinematic"])
            self.assertIn("PREMIUM COMMERCIAL", prompts["premium_commercial"])
            self.assertIn("premium commercial cinematography",
                          prompts["premium_commercial"])
            self.assertIn("handheld UGC review", prompts["viral_ugc"])

    def test_prompt_text_differs_across_energies(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            refs = record["selected_images"]
            prompts = {}
            for energy in ap.AD_ENERGIES:
                plan = styled_plan("hybrid_ugc_cinematic", energy,
                                   shots=["human_ugc", "product_showcase",
                                          "hybrid"])
                prompts[energy] = ap._build_clip_prompt(
                    plan, plan["clips"][0], record, refs, 3)
            self.assertEqual(len(set(prompts.values())), 3)
            self.assertIn("Fast pacing", prompts["viral"])
            self.assertIn("conversational pacing", prompts["natural"])
            self.assertIn("Deliberate pacing", prompts["premium"])

    def test_prompt_includes_shot_camera_balance_and_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = styled_plan("hybrid_ugc_cinematic", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            prompt = ap._build_clip_prompt(
                plan, plan["clips"][1], record, record["selected_images"], 3)
        self.assertIn("Shot type: product_showcase", prompt)
        self.assertIn("Camera: Slow dolly in", prompt)
        self.assertIn("Cinematic product-only beat", prompt)
        self.assertIn("AUDIO / VOICE-OVER:", prompt)
        self.assertIn("no lip-sync", prompt)
        self.assertIn("ENERGY (viral)", prompt)

    def test_fingerprint_changes_with_style_and_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            base = styled_plan("hybrid_ugc_cinematic", "viral",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            other_style = styled_plan("premium_commercial", "viral",
                                      shots=["human_ugc", "product_showcase",
                                             "hybrid"])
            other_energy = styled_plan("hybrid_ugc_cinematic", "premium",
                                       shots=["human_ugc", "product_showcase",
                                              "hybrid"])
            fp_base = ap.plan_fingerprint(base, record)
            self.assertNotEqual(fp_base, ap.plan_fingerprint(other_style,
                                                             record))
            self.assertNotEqual(fp_base, ap.plan_fingerprint(other_energy,
                                                             record))

    def test_prompt_files_are_written_with_selected_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = styled_plan("storytelling", "natural",
                               shots=["human_ugc", "product_showcase",
                                      "hybrid"])
            result = ap.write_prompt_v2(tmp, record, plan, overwrite=True)
            with open(os.path.join(tmp, "prompt", "1.txt"),
                      encoding="utf-8") as f:
                text = f.read()
            ad_plan = json.load(open(result["ad_plan_path"],
                                     encoding="utf-8"))
        self.assertIn("STORYTELLING", text)
        self.assertIn("ENERGY (natural)", text)
        self.assertEqual(ad_plan["ad_style"], "storytelling")
        self.assertEqual(ad_plan["ad_energy"], "natural")
        self.assertEqual(ad_plan["plan_fingerprint"],
                         ap.plan_fingerprint(plan, record))


class PlanConfigResumeTest(unittest.TestCase):
    BASE = {"ad_style": "hybrid_ugc_cinematic", "ad_energy": "viral",
            "target_duration_seconds": 30.0, "language": "en",
            "clip_count": 6}

    def _problem(self, plan):
        return main.plan_config_problem(
            plan, target_duration=30.0, style="hybrid_ugc_cinematic",
            energy="viral", language="en", clip_count=6)

    def test_matching_plan_is_reused(self):
        self.assertIsNone(self._problem(dict(self.BASE)))

    def test_changed_style_is_rejected(self):
        problem = self._problem(dict(self.BASE, ad_style="viral_ugc"))
        self.assertIn("style", problem)

    def test_changed_energy_is_rejected(self):
        problem = self._problem(dict(self.BASE, ad_energy="premium"))
        self.assertIn("energy", problem)

    def test_legacy_plan_without_energy_is_rejected(self):
        legacy = {"style": "ugc_product_showcase",
                  "target_duration_seconds": 30.0, "language": "en",
                  "clip_count": 6}
        problem = self._problem(legacy)
        self.assertIn("ugc_product_showcase", problem)
        self.assertIn("(none)", problem)
        self.assertIn("--overwrite", problem)

    def test_changed_duration_language_or_count_is_rejected(self):
        self.assertIn("duration", self._problem(dict(self.BASE,
                                                     target_duration_seconds=15.0)))
        self.assertIn("language", self._problem(dict(self.BASE,
                                                     language="vi")))
        self.assertIn("clip count", self._problem(dict(self.BASE,
                                                       clip_count=3)))


class AutoAdBackwardCompatibilityTest(unittest.TestCase):
    def test_normal_pipeline_args_are_unchanged(self):
        args = main.parse_args(["a cinematic shot of a city"])
        self.assertFalse(args.auto_ad)
        self.assertIsNone(main.validate_cli_combination(args))
        self.assertEqual(args.video_model, "ltx")
        self.assertEqual(args.aspect, "16:9")

    def test_ad_flags_are_inert_without_auto_ad(self):
        args = main.parse_args(["a cinematic shot of a city",
                                "--ad-style", "premium_commercial",
                                "--ad-energy", "premium"])
        self.assertFalse(args.auto_ad)
        self.assertIsNone(main.validate_cli_combination(args))

    def test_legacy_plan_still_validates_without_new_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = make_record(tmp)
            plan = ap.validate_plan(
                base_plan(), record, target_duration=15,
                style="ugc_product_showcase", language="en", clip_count=3,
                clip_duration=5.0, model="qwen3:8b")
        self.assertEqual(plan["ad_style"], "ugc_product_showcase")
        self.assertIn(plan["ad_energy"], ap.AD_ENERGIES)
        self.assertTrue(all(c["shot_type"] for c in plan["clips"]))


if __name__ == "__main__":
    unittest.main()
