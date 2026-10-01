"""Unit tests: research modes, provider parsing/failures, persistence, CLI."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import main
import ollama as ol
import storyboard as sb
from providers.errors import UnknownModelError
from providers.registry import create_provider, lookup
from providers.research.base import (
    ResearchError,
    ResearchProvider,
    ResearchResult,
    ResearchSource,
)
from providers.research.web import WebResearchProvider

AI_VIDEO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WIKI_SEARCH_FIXTURE = {
    "query": {"search": [{"title": "Humanoid robot"}, {"title": "Atlas (robot)"}]}
}
WIKI_EXTRACT_FIXTURE = {
    "query": {"pages": {
        "1": {"title": "Humanoid robot",
              "extract": "A humanoid robot resembles the human body in shape."},
        "2": {"title": "Atlas (robot)", "extract": ""},
    }}
}
DDG_FIXTURE = {
    "Heading": "Humanoid robot", "AbstractText": "Robots with human form.",
    "AbstractURL": "https://example.com/humanoid", "AbstractSource": "Example",
    "RelatedTopics": [{"Text": "Atlas by Boston Dynamics",
                       "FirstURL": "https://example.com/atlas"}],
}


class NoneModeTest(unittest.TestCase):
    def test_none_returns_empty_without_provider(self):
        import main as main_mod
        real = main_mod.create_provider

        def boom(*args, **kwargs):
            raise AssertionError("research provider must not be created")

        main_mod.create_provider = boom
        try:
            context, meta = main.maybe_research("none", "anything")
        finally:
            main_mod.create_provider = real
        self.assertIsNone(context)
        self.assertIsNone(meta)


class WebModeTest(unittest.TestCase):
    def test_web_invokes_provider(self):
        import main as main_mod
        calls = []
        fixture = ResearchResult(query="q", sources=[
            ResearchSource(title="T", url="https://example.com/x", snippet="S",
                           published="2026-01-01")])

        class StubProvider:
            def research(self, topic):
                calls.append(topic)
                return fixture

        real = main_mod.create_provider
        main_mod.create_provider = lambda *a, **k: StubProvider()
        try:
            context, meta = main.maybe_research("web", "robots")
        finally:
            main_mod.create_provider = real
        self.assertEqual(calls, ["robots"])
        self.assertIn("https://example.com/x", context)
        self.assertIn("T", context)
        self.assertEqual(meta["mode"], "web")
        self.assertEqual(meta["sources"],
                         [{"title": "T", "url": "https://example.com/x",
                           "published": "2026-01-01"}])

    def test_unknown_research_model(self):
        with self.assertRaises(UnknownModelError):
            lookup(AI_VIDEO_DIR, "research", "deep")
        with self.assertRaises(UnknownModelError):
            main.maybe_research("deep", "robots")

    def test_create_web_provider_via_registry(self):
        provider = create_provider("research", AI_VIDEO_DIR, "web")
        self.assertIsInstance(provider, WebResearchProvider)
        self.assertIsInstance(provider, ResearchProvider)


class StoryboardContextTest(unittest.TestCase):
    def test_context_appended_as_separate_message(self):
        import ollama as ol_mod
        captured = {}
        real = ol_mod._post

        def fake_post(url, payload, timeout):
            captured.update(payload)
            return {"message": {"content": '{"scenes": []}'}, "model": "m"}

        ol_mod._post = fake_post
        try:
            ol.generate_storyboard("robots", model="m", num_scenes=1,
                                   research_context="CTX: fact one.")
        except Exception:
            pass
        finally:
            ol_mod._post = real
        messages = captured["messages"]
        self.assertEqual(len(messages), 3)
        self.assertIn("robots", messages[1]["content"])
        self.assertNotIn("CTX", messages[1]["content"])  # original untouched
        self.assertIn("CTX: fact one.", messages[2]["content"])

    def test_no_context_two_messages(self):
        import ollama as ol_mod
        captured = {}
        real = ol_mod._post

        def fake_post(url, payload, timeout):
            captured.update(payload)
            return {"message": {"content": '{"scenes": []}'}, "model": "m"}

        ol_mod._post = fake_post
        try:
            ol.generate_storyboard("robots", model="m", num_scenes=1)
        except Exception:
            pass
        finally:
            ol_mod._post = real
        self.assertEqual(len(captured["messages"]), 2)


class ProviderParsingTest(unittest.TestCase):
    def _provider_with(self, mapping):
        provider = WebResearchProvider(
            name="web", settings={"max_sources": 5, "timeout": 5,
                                  "snippet_chars": 200})
        import providers.research.web as web_mod
        real_json = web_mod.WebResearchProvider._get_json
        real_text = web_mod.WebResearchProvider._get_text
        web_mod.WebResearchProvider._get_json = lambda self, url: mapping(url)
        web_mod.WebResearchProvider._get_text = lambda self, url: ""
        self.addCleanup(setattr, web_mod.WebResearchProvider, "_get_json", real_json)
        self.addCleanup(setattr, web_mod.WebResearchProvider, "_get_text", real_text)
        return provider

    def test_wikipedia_sources(self):
        def mapping(url):
            return WIKI_EXTRACT_FIXTURE if "extracts" in url or "titles" in url \
                else WIKI_SEARCH_FIXTURE
        provider = self._provider_with(mapping)
        result = provider.research("humanoid robots")
        self.assertEqual(result.query, "humanoid robots")
        self.assertEqual(len(result.sources), 1)  # empty extract skipped
        src = result.sources[0]
        self.assertEqual(src.title, "Humanoid robot")
        self.assertTrue(src.url.startswith("https://en.wikipedia.org/wiki/"))
        self.assertIn("resembles", src.snippet)
        self.assertEqual(src.metadata.get("source"), "wikipedia")

    def test_network_failure_raises(self):
        import providers.research.web as web_mod
        provider = WebResearchProvider(name="web", settings={})
        orig_get = web_mod.WebResearchProvider._get_json
        orig_text = web_mod.WebResearchProvider._get_text
        try:
            def failing_get(self, url):
                raise ResearchError("unreachable (simulated)")
            web_mod.WebResearchProvider._get_json = failing_get
            web_mod.WebResearchProvider._get_text = failing_get
            with self.assertRaises(ResearchError):
                provider.research("humanoid robots")
        finally:
            web_mod.WebResearchProvider._get_json = orig_get
            web_mod.WebResearchProvider._get_text = orig_text

    def test_zero_sources_raises(self):
        import providers.research.web as web_mod
        provider = WebResearchProvider(name="web", settings={})
        orig_wiki = web_mod.WebResearchProvider._wikipedia
        orig_ddg = web_mod.WebResearchProvider._duckduckgo
        orig_arxiv = web_mod.WebResearchProvider._arxiv
        orig_hn = web_mod.WebResearchProvider._hn
        try:
            web_mod.WebResearchProvider._wikipedia = lambda self, topic: []
            web_mod.WebResearchProvider._duckduckgo = lambda self, topic: []
            web_mod.WebResearchProvider._arxiv = lambda self, topic: []
            web_mod.WebResearchProvider._hn = lambda self, topic: []
            with self.assertRaises(ResearchError) as ctx:
                provider.research("humanoid robots")
            self.assertIn("no usable sources", str(ctx.exception))
        finally:
            web_mod.WebResearchProvider._wikipedia = orig_wiki
            web_mod.WebResearchProvider._duckduckgo = orig_ddg
            web_mod.WebResearchProvider._arxiv = orig_arxiv
            web_mod.WebResearchProvider._hn = orig_hn

    def test_empty_topic_rejected(self):
        with self.assertRaises(ResearchError):
            WebResearchProvider(name="web", settings={}).research("  ")

    def test_context_rendering_truncates(self):
        result = ResearchResult(query="q", sources=[
            ResearchSource(title=f"T{i}", url=f"https://e.com/{i}",
                           snippet="x" * 500) for i in range(10)])
        text = result.render_context(max_chars=800)
        self.assertIn("T0", text)
        self.assertNotIn("T9", text)
        self.assertIn("Do not invent", text)


class PersistenceTest(unittest.TestCase):
    def test_save_and_load_research(self):
        scenes = sb.validate_storyboard({"scenes": [{
            "id": 1, "duration": 5, "image_prompt": "p", "video_prompt": "v",
            "narration": "n"}]})
        meta = {"mode": "web", "sources": [
            {"title": "T", "url": "https://e.com", "published": None}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sb.json")
            sb.save_storyboard(scenes, path, None, meta)
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw["research"], meta)
            self.assertEqual(sb.load_research(path), meta)
            self.assertEqual(len(sb.load_storyboard(path)), 1)

    def test_old_file_without_research(self):
        scenes = [{"id": 1, "duration": 5, "image_prompt": "p",
                   "video_prompt": "v", "narration": "n"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"scenes": scenes}, f)
            self.assertIsNone(sb.load_research(path))
            self.assertEqual(len(sb.load_storyboard(path)), 1)


class CliResearchTest(unittest.TestCase):
    def test_default_none(self):
        self.assertEqual(main.parse_args(["hello"]).research, "none")

    def test_web_parses(self):
        self.assertEqual(main.parse_args(["hello", "--research", "web"]).research,
                         "web")

    def test_invalid_rejected(self):
        with self.assertRaises(SystemExit):
            main.parse_args(["hello", "--research", "deep"])


if __name__ == "__main__":
    unittest.main()
