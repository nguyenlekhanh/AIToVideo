"""Unit tests: visual-generation gate (no readable text/split/infographic).

No GPU/network: Ollama transport (ol._post) is stubbed. Covers the
deterministic validator, the bounded visual rewrite loop, the rewrite
call itself, and the planner prompt contract.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import main
import ollama as ol
import storyboard as sb


def scene(**over):
    base = {"id": 1, "duration": 5,
            "image_prompt": "A calm office.",
            "video_prompt": "Camera drifts slowly.",
            "narration": "Workers talk."}
    base.update(over)
    return base


CLEAN_IMAGE = (
    "A working-age American office employee sitting alone at a desk in a "
    "modern office, looking thoughtfully out a large window, subtle "
    "concern in their expression, coworkers working naturally in the "
    "distant background, soft natural daylight, photorealistic premium "
    "documentary photography, eye-level camera, medium shot.")


class VisualValidatorTest(unittest.TestCase):
    def test_screen_with_statistic_rejected(self):
        s = scene(image_prompt=(
            "A worker at a desk, computer screen showing "
            "'Job Growth 528,000' in bold letters."))
        errors = sb.validate_visual_prompts([s])
        self.assertEqual(len(errors), 1)
        self.assertIn("Scene #1 image_prompt", errors[0])
        self.assertIn("Research facts must remain in narration/metadata",
                      errors[0])

    def test_sign_with_words_rejected(self):
        s = scene(image_prompt=(
            "Worker standing at a bus stop holding a 'Job Loss' sign."))
        errors = sb.validate_visual_prompts([s])
        self.assertEqual(len(errors), 1)
        self.assertIn("Scene #1 image_prompt", errors[0])

    def test_notice_rejected(self):
        s = scene(image_prompt=(
            "Worker sitting at a desk with a 'Wage Growth Slows' notice, "
            "surrounded by fragmented news headlines."))
        self.assertTrue(sb.validate_visual_prompts([s]))

    def test_split_screen_rejected(self):
        s = scene(image_prompt=(
            "Split-screen comparison of employment then and now, "
            "two panels side by side."))
        errors = sb.validate_visual_prompts([s])
        self.assertTrue(any("multi-panel" in e for e in errors))

    def test_infographic_rejected(self):
        s = scene(image_prompt=(
            "An infographic layout explaining unemployment with charts."))
        self.assertTrue(sb.validate_visual_prompts([s]))

    def test_two_sided_composition_rejected(self):
        s = scene(image_prompt=(
            "On one side a busy office, on the other side an empty "
            "factory floor."))
        errors = sb.validate_visual_prompts([s])
        self.assertTrue(any("multi-panel" in e for e in errors))

    def test_video_prompt_checked(self):
        s = scene(video_prompt=(
            "Statistics float across the screen as charts animate."))
        errors = sb.validate_visual_prompts([s])
        self.assertTrue(any("video_prompt" in e for e in errors))

    def test_clean_prompt_with_constraints_passes(self):
        s = scene(image_prompt=CLEAN_IMAGE)
        self.assertEqual(sb.validate_visual_prompts([s]), [])

    def test_bare_clean_prompt_passes(self):
        s = scene(image_prompt=(
            "worker sitting at a desk in a modern office looking out "
            "a window"))
        self.assertEqual(sb.validate_visual_prompts([s]), [])

    def test_bare_monitor_rejected(self):
        s = scene(image_prompt=(
            "worker looking at an out-of-focus computer monitor"))
        errors = sb.validate_visual_prompts([s])
        self.assertTrue(any("information-display" in e for e in errors))

    def test_bare_newspaper_rejected(self):
        s = scene(image_prompt=(
            "man reading a newspaper on a train"))
        self.assertTrue(sb.validate_visual_prompts([s]))

    def test_bare_sign_rejected(self):
        s = scene(image_prompt=(
            "worker holding a sign at a bus stop"))
        self.assertTrue(sb.validate_visual_prompts([s]))

    def test_laptop_bag_allowed(self):
        s = scene(image_prompt=(
            "employee carrying a laptop bag through an office entrance, "
            "morning daylight"))
        self.assertEqual(sb.validate_visual_prompts([s]), [])

    def test_closed_laptop_allowed(self):
        s = scene(image_prompt=(
            "professional sitting at a kitchen table, laptop closed "
            "beside them, hands resting on the table"))
        self.assertEqual(sb.validate_visual_prompts([s]), [])

    def test_overloaded_prompt_rejected(self):
        s = scene(image_prompt=" ".join(["thoughtful office worker"] * 70))
        errors = sb.validate_visual_prompts([s])
        self.assertTrue(any("overloaded" in e for e in errors))

    def test_concise_prompt_passes_length(self):
        words = CLEAN_IMAGE.split()
        self.assertLessEqual(len(words), 80)

    def test_narration_never_inspected(self):
        s = scene(narration="Unemployment hit 'record 5.2%' this month, "
                            "split-screen coverage everywhere.")
        self.assertEqual(sb.validate_visual_prompts([s]), [])

    def test_scene_number_reported(self):
        scenes = [scene(), scene(image_prompt="A collage of workplaces.")]
        errors = sb.validate_visual_prompts(scenes)
        self.assertEqual(len(errors), 1)
        self.assertIn("Scene #2", errors[0])


class VisualRewriteLoopTest(unittest.TestCase):
    def test_fixes_in_one_round_preserving_grounding(self):
        calls = []

        def rewrite(scn, reason):
            calls.append(reason)
            fixed = dict(scn)
            fixed["image_prompt"] = "A calm office, single continuous frame."
            fixed["video_prompt"] = "Camera drifts slowly."
            return fixed

        scenes = [dict(scene(image_prompt="Screen showing '5.2% jobs'.",
                             research_fact_ids=["fact_001"],
                             source_ids=["src_001"],
                             grounding="researched"),
                       id=1)]
        out, rewrites = main.ensure_clean_visuals(scenes, rewrite)
        self.assertEqual(rewrites, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("Scene #1", calls[0])
        self.assertEqual(out[0]["narration"], "Workers talk.")
        self.assertEqual(out[0]["research_fact_ids"], ["fact_001"])
        self.assertEqual(out[0]["source_ids"], ["src_001"])
        self.assertEqual(out[0]["grounding"], "researched")
        self.assertEqual(out[0]["id"], 1)

    def test_persistent_failure_raises_bounded(self):
        calls = []

        def rewrite(scn, reason):
            calls.append(reason)
            return dict(scn)  # never improves

        bad = [scene(image_prompt="An infographic of jobs.")]
        with self.assertRaisesRegex(
                ValueError, "still request rendered text/statistics"):
            main.ensure_clean_visuals(bad, rewrite, max_rounds=2)
        self.assertEqual(len(calls), 2)

    def test_clean_scenes_zero_rewrites(self):
        scenes = [scene(image_prompt=CLEAN_IMAGE)]
        out, rewrites = main.ensure_clean_visuals(
            scenes, lambda s, r: s)
        self.assertEqual(rewrites, 0)
        self.assertEqual(out, scenes)


class RewriteVisualsCallTest(unittest.TestCase):
    def test_rewrites_only_prompts(self):
        import ollama as ol_mod
        fixed = {"image_prompt": "A calm office, single continuous frame.",
                 "video_prompt": "Camera drifts slowly."}
        real = ol_mod._post
        ol_mod._post = lambda url, payload, timeout: {
            "message": {"content": json.dumps(fixed)}}
        try:
            original = scene(image_prompt="Screen showing '5.2%'.",
                             research_fact_ids=["fact_001"],
                             source_ids=["src_001"],
                             grounding="researched")
            updated = ol.rewrite_scene_visuals(
                original, "quoted text", model="m")
        finally:
            ol_mod._post = real
        self.assertEqual(updated["image_prompt"], fixed["image_prompt"])
        self.assertEqual(updated["video_prompt"], fixed["video_prompt"])
        for key in ("id", "duration", "narration", "research_fact_ids",
                    "source_ids", "grounding"):
            self.assertEqual(updated[key], original[key])

    def test_malformed_json_raises(self):
        import ollama as ol_mod
        real = ol_mod._post
        ol_mod._post = lambda url, payload, timeout: {
            "message": {"content": "not json"}}
        try:
            with self.assertRaises(RuntimeError):
                ol.rewrite_scene_visuals(scene(), "reason", model="m")
        finally:
            ol_mod._post = real

    def test_empty_prompts_raise(self):
        import ollama as ol_mod
        real = ol_mod._post
        ol_mod._post = lambda url, payload, timeout: {
            "message": {"content": '{"image_prompt": "", "video_prompt": ""}'}}
        try:
            with self.assertRaises(RuntimeError):
                ol.rewrite_scene_visuals(scene(), "reason", model="m")
        finally:
            ol_mod._post = real


class PlannerPromptContractTest(unittest.TestCase):
    def test_system_prompt_has_visual_rules(self):
        prompt = ol.SYSTEM_PROMPT
        for required in (
                "ONLY what a physical camera can see",
                "no screens, monitors, computers",
                "Research facts belong in narration/metadata ONLY",
                "Target 40-80 words",
                "animate the SAME single scene",
                "Do not append lists of negative concepts"):
            with self.subTest(required=required[:30]):
                self.assertIn(required, prompt)

    def test_research_rules_intact(self):
        prompt = ol.SYSTEM_PROMPT
        for required in ("research_fact_ids", "do not invent facts",
                         "grounding", "Exactly the requested number"):
            with self.subTest(required=required):
                self.assertIn(required, prompt)


if __name__ == "__main__":
    unittest.main()
