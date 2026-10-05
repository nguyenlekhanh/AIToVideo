"""Regression tests: current-news research gate + targeted retry.

No network: provider fetchers (_arxiv/_hn/_wikipedia/_duckduckgo) are
stubbed per test. Ollama planner (ol.generate_storyboard) is stubbed
where the storyboard stage is reached, and asserted UNCALLED when the
gate must stop the pipeline before planning.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
import ollama as ol
import state as st
import storyboard as sb
from providers.research.base import ResearchError, ResearchSource
from providers.research.web import WebResearchProvider
from providers.research import pipeline as pipe

JOBS_TOPIC = ("America's Job Market Looks Strong - So Why Are Workers "
              "Feeling Less Secure?")


def wiki_article(title="Unemployment in the United States"):
    return ResearchSource(
        title=title, url="https://en.wikipedia.org/wiki/" + title.replace(
            " ", "_"),
        snippet=("Unemployment in the United States discusses the causes "
                 "and measurement of unemployment across the labor market "
                 "over many decades of economic history."),
        publisher="Wikipedia", published_at=None, source_type="encyclopedic",
        metadata={"source": "wikipedia"})


def hn_hit(title, url, date):
    return ResearchSource(
        title=title, url=url,
        snippet=(f"Hacker News discussion (10 points, 5 comments): {title}"),
        publisher="example.com", published_at=date, source_type="technical",
        metadata={"source": "hackernews"})


RECENT_HITS = [
    hn_hit("U.S. Unemployment Rate Rises to 4.4 Percent in December",
           "https://example.com/jobs1", "2026-08-07"),
    hn_hit("U.S. Employers Added Fewer Jobs Than Expected in March",
           "https://example.com/jobs2", "2026-03-06"),
]


def patch_fetchers(testcase, *, wiki=None, arxiv=None, hn=None, ddg=None):
    import providers.research.web as web_mod
    real = {}
    for name, func in (("_wikipedia", wiki), ("_arxiv", arxiv),
                       ("_hn", hn), ("_duckduckgo", ddg)):
        real[name] = getattr(web_mod.WebResearchProvider, name)
        if func is not None:
            setattr(web_mod.WebResearchProvider, name, func)
    testcase.addCleanup(
        lambda: [setattr(web_mod.WebResearchProvider, n, f)
                 for n, f in real.items()])


def background_only_fetchers():
    return {
        "wiki": lambda self, topic: [wiki_article()],
        "arxiv": lambda self, topic: [],
        "hn": lambda self, topic: [],
        "ddg": lambda self, topic: [],
    }


def retry_aware_fetchers():
    """Pass 1 (original long topic): background only. Retry queries
    (short targeted strings): dated hits."""
    def hn(self, topic):
        if topic == JOBS_TOPIC:
            return []
        return list(RECENT_HITS)
    return {
        "wiki": lambda self, topic: [wiki_article()],
        "arxiv": lambda self, topic: [],
        "hn": hn,
        "ddg": lambda self, topic: [],
    }


def creative_board():
    return {"subject": {"type": "none", "identity": "", "features": "",
                        "clothing_or_equipment": "", "consistency": ""},
            "scenes": [
                {"id": 1, "duration": 5, "image_prompt": "p",
                 "video_prompt": "v", "narration": "Workers share stories.",
                 "grounding": "creative"},
                {"id": 2, "duration": 5, "image_prompt": "p",
                 "video_prompt": "v", "narration": "A calm overview.",
                 "grounding": "creative"},
            ]}


def fresh_ctx(tmp, project="testproj"):
    import os
    state = st.new_state(project)
    return {"project": project, "base_dir": tmp, "base_seed": 1,
            "state": state, "state_path": os.path.join(tmp, "state.json"),
            "storyboard_path": os.path.join(tmp, "storyboard.json")}


def fresh_args(topic):
    return SimpleNamespace(research="web", prompt=topic, keyframes=None,
                           script=None, scenes=None, character_reference=None)


class NewsIntentTest(unittest.TestCase):
    def test_jobs_topic_is_current_news_without_latest(self):
        self.assertEqual(pipe.detect_news_intent(JOBS_TOPIC), "current_news")

    def test_evergreen_stays_background(self):
        self.assertEqual(
            pipe.detect_news_intent("How does unemployment insurance work?"),
            "background")

    def test_explicit_markers_are_current_news(self):
        self.assertEqual(
            pipe.detect_news_intent("latest breakthroughs in robotics"),
            "current_news")

    def test_plain_topic_is_background(self):
        self.assertEqual(pipe.detect_news_intent("humanoid robots"),
                         "background")

    def test_domain_topics(self):
        for topic in ("Federal Reserve rate decision looms",
                      "Supreme Court developments this term",
                      "stock market selloff continues",
                      "inflation eases slightly",
                      "Congress debates the budget bill",
                      "election results spark reactions"):
            with self.subTest(topic=topic):
                self.assertEqual(pipe.detect_news_intent(topic),
                                 "current_news")

    def test_explainer_with_marker_stays_current(self):
        self.assertEqual(
            pipe.detect_news_intent("How will the latest Fed decision "
                                    "affect markets?"),
            "current_news")


class TargetedQueriesTest(unittest.TestCase):
    def test_jobs_templates_and_dynamic_year(self):
        import datetime
        year = str(datetime.date.today().year)
        queries = pipe.targeted_queries(JOBS_TOPIC)
        self.assertIn(f"U.S. jobs report {year}", queries)
        self.assertIn(f"U.S. unemployment rate {year}", queries)
        self.assertLessEqual(len(queries), pipe.MAX_RETRY_QUERIES)
        # No hardcoded stale year anywhere.
        for query in queries:
            for token in query.split():
                if token.isdigit() and len(token) == 4:
                    self.assertEqual(token, year)

    def test_generic_topic_falls_back(self):
        queries = pipe.targeted_queries(
            "breaking developments in downtown transit")
        self.assertTrue(len(queries) >= 1)

    def test_empty_topic(self):
        self.assertEqual(pipe.targeted_queries("  "), [])


class GateUnitTest(unittest.TestCase):
    def test_high_passes_others_fail(self):
        import providers.research.base as base
        for quality, expected in (("high", True), ("medium", False),
                                  ("background_only", False),
                                  ("insufficient", False)):
            pack = base.ResearchPack(query="q", research_quality=quality,
                                     retrieved_at="t")
            with self.subTest(quality=quality):
                self.assertEqual(pipe.current_evidence_sufficient(pack),
                                 expected)


class RetryFlowTest(unittest.TestCase):
    def test_background_only_single_pass_for_evergreen(self):
        calls = []
        fetchers = background_only_fetchers()
        orig_hn = fetchers["hn"]

        def counting_hn(self, topic):
            calls.append(topic)
            return orig_hn(self, topic)

        fetchers["hn"] = counting_hn
        patch_fetchers(self, **fetchers)
        provider = WebResearchProvider(name="web", settings={})
        pack_result = provider.research("How does unemployment insurance work?")
        self.assertEqual(pack_result.pack.research_quality, "background_only")
        # No retry: dated fetchers ran once each (pass 1 only).
        self.assertEqual(len(calls), 1)

    def test_retry_reaches_high(self):
        patch_fetchers(self, **retry_aware_fetchers())
        provider = WebResearchProvider(name="web", settings={})
        result = provider.research(JOBS_TOPIC)
        self.assertEqual(result.pack.research_quality, "high")
        recent = [f for f in result.pack.facts
                  if f.freshness in ("current", "recent")]
        self.assertGreaterEqual(len(recent), 2)
        distinct = {s for f in recent for s in f.source_ids}
        self.assertGreaterEqual(len(distinct), 2)

    def test_failed_retry_raises_before_planner(self):
        patch_fetchers(self, **background_only_fetchers())
        provider = WebResearchProvider(name="web", settings={})
        with self.assertRaisesRegex(ResearchError, "current-news"):
            provider.research(JOBS_TOPIC)


class PlannerGatingTest(unittest.TestCase):
    def _run_flow(self, topic, fetchers, planner_board=None):
        import tempfile
        import os
        patch_fetchers(self, **fetchers)
        planner_calls = []
        real_gen = ol.generate_storyboard

        def fake_gen(*args, **kwargs):
            planner_calls.append((args, kwargs))
            return planner_board or creative_board()

        ol.generate_storyboard = fake_gen
        self.addCleanup(setattr, ol, "generate_storyboard", real_gen)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = fresh_ctx(tmp)
            try:
                rc = main.run_fresh_flow(
                    fresh_args(topic), ctx, {},
                    "http://127.0.0.1:11434", "qwen3:8b", 2, ["storyboard"])
            except Exception as exc:
                return exc, planner_calls, tmp
            return rc, planner_calls, tmp

    def test_planner_runs_only_after_retry(self):
        # TEST 1 + TEST 6: background pass 1, current pass 2 -> storyboard.
        outcome, planner_calls, _ = self._run_flow(
            JOBS_TOPIC, retry_aware_fetchers())
        self.assertEqual(outcome, 0)
        self.assertEqual(len(planner_calls), 1)

    def test_planner_never_runs_when_retry_fails(self):
        # TEST 2 + TEST 5: background stays background -> clear error.
        outcome, planner_calls, _ = self._run_flow(
            JOBS_TOPIC, background_only_fetchers())
        self.assertIsInstance(outcome, ResearchError)
        self.assertIn("current-news", str(outcome))
        self.assertEqual(planner_calls, [])

    def test_evergreen_passes_through(self):
        # TEST 4: evergreen + background research -> planner runs.
        outcome, planner_calls, _ = self._run_flow(
            "How does unemployment insurance work?",
            background_only_fetchers())
        self.assertEqual(outcome, 0)
        self.assertEqual(len(planner_calls), 1)


class MultiSourceGateTest(unittest.TestCase):
    def test_official_plus_news_passes(self):
        # TEST 3: recent official + recent reputable news -> planner runs.
        official = hn_hit("Bureau of Labor Statistics Jobs Report Shows "
                          "Steady Gains in September",
                          "https://www.bls.gov/news.release/empsit.nr0.htm",
                          "2026-09-04")
        official.source_type = "official"
        official.publisher = "bls.gov"
        news = hn_hit("Reuters Reports U.S. Job Growth Cools in September",
                      "https://www.reuters.com/markets/us/job-growth-sep",
                      "2026-09-15")
        news.source_type = "news"
        news.publisher = "reuters.com"
        patch_fetchers(
            self,
            wiki=lambda self_, topic: [wiki_article()],
            arxiv=lambda self_, topic: [],
            hn=lambda self_, topic: [official, news],
            ddg=lambda self_, topic: [])
        provider = WebResearchProvider(name="web", settings={})
        result = provider.research("U.S. jobs report analysis")
        self.assertEqual(result.pack.research_quality, "high")


if __name__ == "__main__":
    unittest.main()
