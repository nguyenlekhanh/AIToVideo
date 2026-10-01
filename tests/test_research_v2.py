"""Unit tests: query expansion, source scoring, diagnostics (no network)."""
from __future__ import annotations

import unittest

from providers.research import pipeline as pipe
from providers.research.base import ResearchSource


def src(title, snippet, source_type="other", published_at=None):
    return ResearchSource(title=title, url=f"https://example.com/{title[:8]}",
                          snippet=snippet, published_at=published_at,
                          source_type=source_type)


class ExpansionTest(unittest.TestCase):
    def test_temporal_topic_expands(self):
        queries = pipe.expand_queries("latest humanoid robot breakthroughs 2026", 2)
        self.assertEqual(len(queries), 3)
        self.assertTrue(queries[0].startswith("latest"))
        year = queries[1].split()[-1]
        self.assertTrue(year.isdigit() and len(year) == 4)
        self.assertIn("announcement", queries[2])

    def test_non_temporal_single(self):
        self.assertEqual(pipe.expand_queries("a cat sits quietly", 2),
                         ["a cat sits quietly"])
        self.assertEqual(pipe.expand_queries("", 2), [])

    def test_strip_markers(self):
        base = pipe.strip_temporal_markers("latest humanoid robot breakthroughs 2026")
        self.assertNotIn("latest", base.lower())
        self.assertNotIn("2026", base)
        self.assertIn("humanoid", base.lower())

    def test_max_expansions_bound(self):
        self.assertLessEqual(len(pipe.expand_queries("latest robots", 1)), 2)


class ScoringTest(unittest.TestCase):
    def test_order_recent_authoritative_first(self):
        import datetime as _dt
        recent = (_dt.date.today() - _dt.timedelta(days=10)).isoformat()
        old = (_dt.date.today() - _dt.timedelta(days=800)).isoformat()
        q = "humanoid robot breakthrough"
        official_recent = src("Official humanoid robot breakthrough announced",
                              "official humanoid robot breakthrough", "official", recent)
        generic_recent = src("Humanoid robot breakthrough news roundup",
                             "humanoid robot breakthrough news", "other", recent)
        official_old = src("Official humanoid robot program history",
                           "official humanoid robot program", "official", old)
        undated = src("Humanoid robot background article",
                      "humanoid robot background", "encyclopedic", None)
        ranked = pipe.sort_sources_by_score(
            [undated, official_old, generic_recent, official_recent], q)
        self.assertEqual(
            [s.title for s in ranked],
            ["Official humanoid robot breakthrough announced",
             "Humanoid robot breakthrough news roundup",
             "Official humanoid robot program history",
             "Humanoid robot background article"])

    def test_scoring_never_overrides_filtering(self):
        pack = pipe.build_pack("humanoid robots latest breakthroughs", [
            src("China trade tariffs diplomacy", "trade tariffs diplomacy",
                "news", None),
        ])
        self.assertEqual(pack.sources, [])
        self.assertEqual(pack.research_quality, "insufficient")


class DiagnosticsAndIdsTest(unittest.TestCase):
    def test_stable_dense_ids(self):
        pack = pipe.build_pack("robot lab", [
            src("Robot lab opens downtown", "robot lab opens downtown"),
            src("Robot lab funding round", "robot lab funding round"),
        ])
        self.assertEqual([s.id for s in pack.sources], ["src_001", "src_002"])
        pack2 = pipe.build_pack("robot lab", [
            src("Robot lab opens downtown", "robot lab opens downtown"),
            src("Robot lab funding round", "robot lab funding round"),
        ])
        self.assertEqual([s.id for s in pack2.sources], ["src_001", "src_002"])


if __name__ == "__main__":
    unittest.main()
