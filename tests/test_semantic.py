"""Unit tests: deterministic semantic grounding (no models, no network)."""
from __future__ import annotations

import unittest

from providers.research.base import ResearchFact
from providers.research.semantic import (
    STATUS_PARTIAL,
    STATUS_SUPPORTED,
    STATUS_UNKNOWN,
    STATUS_UNSUPPORTED,
    SemanticGrounder,
    TokenOverlapGrounder,
)


def fact(fid, claim, sids=("src_001",)):
    return ResearchFact(id=fid, claim=claim, source_ids=list(sids))


class SpecExamplesTest(unittest.TestCase):
    def setUp(self):
        self.grounder = TokenOverlapGrounder()
        self.assertIsInstance(self.grounder, SemanticGrounder)

    def test_example1_exact_supported(self):
        result = self.grounder.ground(
            "NVIDIA introduced the DGX A100 system.",
            [fact("fact_001", "NVIDIA introduced the DGX A100 system in 2020.")])
        self.assertEqual(result.status, STATUS_SUPPORTED)
        self.assertEqual(result.fact_ids, ["fact_001"])

    def test_example2_wrong_year_unsupported(self):
        result = self.grounder.ground(
            "NVIDIA introduced the DGX A100 system in 2026.",
            [fact("fact_001", "NVIDIA introduced the DGX A100 system in 2020.")])
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIn("2026", result.reason)

    def test_example3_unrelated_unsupported(self):
        result = self.grounder.ground(
            "NVIDIA's AI hardware has transformed humanoid robotics.",
            [fact("fact_001", "NVIDIA introduced the DGX A100 system.")])
        self.assertIn(result.status, (STATUS_UNSUPPORTED, STATUS_PARTIAL))

    def test_example4_paraphrase_supported(self):
        result = self.grounder.ground(
            "A humanoid robot performed warehouse tasks.",
            [fact("fact_001",
                  "A company demonstrated a humanoid robot performing warehouse tasks.")])
        self.assertEqual(result.status, STATUS_SUPPORTED)

    def test_example5_overclaim_unsupported(self):
        result = self.grounder.ground(
            "Humanoid robots are now replacing most warehouse workers.",
            [fact("fact_001",
                  "A company demonstrated a humanoid robot performing warehouse tasks.")])
        self.assertEqual(result.status, STATUS_UNSUPPORTED)


class RequirementTest(unittest.TestCase):
    def setUp(self):
        self.grounder = TokenOverlapGrounder()

    def test_wrong_number_unsupported(self):
        result = self.grounder.ground(
            "The robot has 31 joints and lifts 70 kilograms.",
            [fact("fact_001", "The robot has 24 joints and lifts 70 kilograms.")])
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIn("31", result.reason)

    def test_matching_numbers_supported(self):
        result = self.grounder.ground(
            "The robot has 24 joints.",
            [fact("fact_001", "The robot has 24 joints and lifts 70 kilograms.")])
        self.assertEqual(result.status, STATUS_SUPPORTED)

    def test_unsupported_superlative(self):
        result = self.grounder.ground(
            "This is the fastest humanoid robot ever built.",
            [fact("fact_001", "The humanoid robot walked across the room.")])
        self.assertIn(result.status, (STATUS_UNSUPPORTED, STATUS_PARTIAL))

    def test_partial_coverage(self):
        result = self.grounder.ground(
            "Atlas can jump and fly over tall buildings.",
            [fact("fact_001", "Atlas can jump.")])
        self.assertEqual(result.status, STATUS_PARTIAL)
        self.assertGreaterEqual(result.score, 0.35)
        self.assertLess(result.score, 0.70)

    def test_unrelated_unsupported(self):
        result = self.grounder.ground(
            "The stock market reached record highs on Tuesday.",
            [fact("fact_001", "Atlas can jump.")])
        self.assertEqual(result.status, STATUS_UNSUPPORTED)

    def test_missing_facts_unknown(self):
        result = self.grounder.ground("Atlas can jump.", [])
        self.assertEqual(result.status, STATUS_UNKNOWN)
        self.assertEqual(result.score, 0.0)

    def test_new_proper_entity_unsupported(self):
        result = self.grounder.ground(
            "Boston Dynamics engineers celebrated the launch.",
            [fact("fact_001", "Engineers celebrated the launch.")])
        self.assertEqual(result.status, STATUS_UNSUPPORTED)
        self.assertIn("Boston Dynamics", result.reason)

    def test_score_and_reason_present(self):
        result = self.grounder.ground(
            "Atlas can jump.", [fact("fact_001", "Atlas can jump high.")])
        self.assertEqual(result.status, STATUS_SUPPORTED)
        self.assertGreater(result.score, 0.7)
        self.assertTrue(result.reason)


if __name__ == "__main__":
    unittest.main()
