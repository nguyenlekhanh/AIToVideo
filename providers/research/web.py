"""Keyless web research provider (stdlib only, no credentials).

Source priority for current/latest claims (Phase 2):
1. research papers / preprints (arXiv API: dated, authoritative)
2. technical press/discussion (Hacker News Algolia API: dated)
3. encyclopedic background (Wikipedia API: reliable, undated)
4. other (DuckDuckGo instant-answer: best-effort, often empty)

Rationale: the local Ollama server (probed: no /api/web_search,
/api/web_fetch -> 404) offers no usable web-search mechanism, and the
DuckDuckGo HTML endpoint returns a bot-filtered shell, so HTML scraping
is not viable. Freshness-sensitive queries needing live news still need
a keyed search API -- add a new provider + optional *_API_KEY env var
for that; do not pretend these sources are live news.

Raises ResearchError (never fabricated content) when the network fails
or when no source yields anything usable.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as _ET

from . import pipeline as _pipe
from .base import (
    ResearchError,
    ResearchPackResult,
    ResearchProvider,
    ResearchResult,
    ResearchSource,
)


class WebResearchProvider(ResearchProvider):
    provider_id = "web"

    def __init__(self, name: str = "web", settings: dict | None = None, **kwargs):
        self.name = name
        self.settings = settings or {}
        self.max_sources = int(self.settings.get("max_sources", 5))
        self.timeout = int(self.settings.get("timeout", 20))
        self.snippet_chars = int(self.settings.get("snippet_chars", 1200))
        self.max_facts = int(self.settings.get("max_facts", 8))
        self.relevance_threshold = float(self.settings.get("relevance_threshold", 0.15))
        self.max_expansions = int(self.settings.get("max_expansions", 2))
        self.user_agent = str(self.settings.get("user_agent", "Mozilla/5.0"))

    # -- HTTP (stdlib only) --
    def _get_json(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        return self._fetch(req, url, parse_json=True)

    def _get_text(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        return self._fetch(req, url, parse_json=False)

    def _fetch(self, req, url: str, parse_json: bool):
        """GET with bounded retries on rate limiting (Wikimedia throttles
        bursts aggressively). Non-retryable failures raise ResearchError."""
        import time
        import urllib.error
        last: Exception | None = None
        for attempt in (0, 1, 2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                    return json.loads(raw) if parse_json else raw
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in (403, 429, 500, 502, 503) or attempt == 2:
                    break
                time.sleep(2 * (attempt + 1))
            except Exception as exc:
                last = exc
                break
        raise ResearchError(
            f"Web research request failed ({url[:80]}...): {last}. "
            f"Check Internet access; --research none works fully offline.") from last

    # -- sources --
    def _wikipedia(self, topic: str) -> list[ResearchSource]:
        search_q = urllib.parse.urlencode(
            {"action": "query", "format": "json", "list": "search",
             "srsearch": topic, "srlimit": self.max_sources, "srprop": ""})
        found = self._get_json("https://en.wikipedia.org/w/api.php?" + search_q)
        entries = (found.get("query") or {}).get("search", [])
        titles = [e.get("title") for e in entries if e.get("title")][:self.max_sources]
        if not titles:
            return []
        extract_q = urllib.parse.urlencode(
            {"action": "query", "format": "json", "prop": "extracts|revisions",
             "exintro": 1, "explaintext": 1, "redirects": 1,
             "rvprop": "timestamp",
             "titles": "|".join(titles)})
        pages = self._get_json("https://en.wikipedia.org/w/api.php?" + extract_q)
        sources = []
        for page in (pages.get("query") or {}).get("pages", {}).values():
            title = page.get("title", "")
            extract = (page.get("extract") or "").strip()
            if not title or not extract or "missing" in page:
                continue
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
                title.replace(" ", "_"))
            revisions = page.get("revisions") or []
            first_rev = revisions[0] if revisions else {}
            last_revised = (first_rev.get("timestamp", "") or "")[:10] or ""
            sources.append(ResearchSource(
                title=title, url=url, snippet=extract[:self.snippet_chars],
                publisher="Wikipedia", published_at=None,
                source_type="encyclopedic",
                metadata={"source": "wikipedia",
                          "last_revised": last_revised or None}))
        return sources

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return urllib.parse.urlparse(url).netloc.lower() or "unknown"
        except Exception:
            return "unknown"

    def _arxiv(self, topic: str) -> list[ResearchSource]:
        """Research papers via the keyless arXiv API (dated, authoritative)."""
        query = urllib.parse.urlencode(
            {"search_query": f"all:{topic}", "start": 0,
             "max_results": self.max_sources, "sortBy": "submittedDate",
             "sortOrder": "descending"})
        try:
            raw = self._get_text("https://export.arxiv.org/api/query?" + query)
        except ResearchError:
            return []  # best-effort source; others carry the result
        sources = []
        try:
            root = _ET.fromstring(raw)
        except Exception:
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns)[:self.max_sources]:
            title = " ".join((entry.findtext("a:title", "", ns) or "").split())
            summary = " ".join((entry.findtext("a:summary", "", ns) or "").split())
            published = (entry.findtext("a:published", "", ns) or "")[:10] or None
            link = ""
            for element in entry.findall("a:link", ns):
                if element.get("type") == "text/html":
                    link = element.get("href") or ""
            if not title or not link:
                continue
            authors = [a.findtext("a:name", "", ns)
                       for a in entry.findall("a:author", ns)][:3]
            sources.append(ResearchSource(
                title=title, url=link, snippet=summary[:self.snippet_chars],
                publisher="arXiv", published_at=published,
                source_type="research",
                metadata={"source": "arxiv",
                          "authors": [a for a in authors if a]}))
        return sources

    def _hn(self, topic: str) -> list[ResearchSource]:
        """Technical press/discussion via the keyless HN Algolia API (dated)."""
        query = urllib.parse.urlencode(
            {"query": topic, "tags": "story", "hitsPerPage": self.max_sources})
        try:
            data = self._get_json(
                "https://hn.algolia.com/api/v1/search?" + query)
        except ResearchError:
            return []  # best-effort source
        sources = []
        for hit in data.get("hits", [])[:self.max_sources]:
            title = (hit.get("title") or "").strip()
            url = hit.get("url") or ""
            if not url and hit.get("objectID"):
                url = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            if not title or not url:
                continue
            created = (hit.get("created_at") or "")[:10] or None
            sources.append(ResearchSource(
                title=title, url=url,
                snippet=f"Hacker News discussion ({hit.get('points', 0)} points, "
                        f"{hit.get('num_comments', 0)} comments): {title}"[:self.snippet_chars],
                publisher=self._domain(url), published_at=created,
                source_type="technical",
                metadata={"source": "hackernews",
                          "author": hit.get("author", "")}))
        return sources

    def _duckduckgo(self, topic: str) -> list[ResearchSource]:
        query = urllib.parse.urlencode(
            {"q": topic, "format": "json", "no_html": 1, "skip_disambig": 1})
        try:
            data = self._get_json("https://api.duckduckgo.com/?" + query)
        except ResearchError:
            return []  # best-effort source; Wikipedia carries the result
        sources = []
        abstract, url = (data.get("AbstractText") or "").strip(), data.get("AbstractURL") or ""
        if abstract and url:
            sources.append(ResearchSource(
                title=data.get("Heading") or topic, url=url,
                snippet=abstract[:self.snippet_chars],
                publisher=data.get("AbstractSource") or self._domain(url),
                source_type="other",
                metadata={"source": "duckduckgo",
                          "abstract_source": data.get("AbstractSource", "")}))
        for item in data.get("RelatedTopics", []):
            if isinstance(item, dict) and item.get("Text") and item.get("FirstURL"):
                sources.append(ResearchSource(
                    title=(item.get("Text") or "")[:120], url=item["FirstURL"],
                    snippet=(item.get("Text") or "")[:self.snippet_chars],
                    publisher=self._domain(item["FirstURL"]),
                    source_type="other",
                    metadata={"source": "duckduckgo"}))
            if len(sources) >= self.max_sources:
                break
        return sources

    # -- ResearchProvider interface --
    def research(self, topic: str) -> ResearchPackResult:
        if not (topic or "").strip():
            raise ResearchError("Cannot research an empty topic.")
        topic = topic.strip()
        intent = _pipe.detect_news_intent(topic)
        print(f"[Research] intent={intent}")
        collected: list[ResearchSource] = []
        seen_urls: set[str] = set()
        diagnostics: list[str] = []
        pass_summaries: list[str] = []

        def add(new: list[ResearchSource]) -> None:
            for src in new:
                if src.url and src.url not in seen_urls:
                    seen_urls.add(src.url)
                    collected.append(src)

        def attempt(label: str, func, *args) -> None:
            try:
                got = func(*args)
                add(got)
                diagnostics.append(f"{label}: {len(got)} sources")
            except ResearchError as exc:
                # Individual source failure must not kill the run; the
                # failure is recorded, never turned into a success.
                diagnostics.append(f"{label}: failed ({str(exc)[:120]})")

        # Dated fetchers run against every query expansion so temporal
        # topics get several chances at recent sources.
        expansions = _pipe.expand_queries(topic, self.max_expansions)
        diagnostics.append(f"query expansions ({len(expansions)}): "
                           + " | ".join(expansions))
        for qi, query in enumerate(expansions, start=1):
            attempt(f"arxiv[q{qi}]", self._arxiv, query)
            attempt(f"hn[q{qi}]", self._hn, query)
        # Background/context fetchers run once on the original topic.
        attempt("wikipedia", self._wikipedia, topic)
        if len([s for s in collected if s.source_type in ("research", "technical",
                                                          "news", "official")]) < self.max_sources:
            attempt("duckduckgo", self._duckduckgo, topic)
        if not collected:
            raise ResearchError(
                f"Web research found no usable sources for {topic!r}. "
                f"Refine the topic or use --research none. "
                f"Diagnostics: {'; '.join(diagnostics)}")
        pack = _pipe.build_pack(topic, collected, max_facts=self.max_facts,
                                relevance_threshold=self.relevance_threshold)
        pass_summaries.append(f"pass 1: {len(pack.sources)} source(s), "
                              f"{pack.research_quality}")
        print(f"[Research] pass=1 sources={len(pack.sources)} "
              f"quality={pack.research_quality}")
        if intent == _pipe.NEWS_INTENT_CURRENT \
                and not _pipe.current_evidence_sufficient(pack):
            # Gate: a background-only pack must NEVER reach the storyboard
            # planner for a current-news topic. Retry with short targeted
            # queries (dated fetchers only) before giving up.
            print("[Research] current-news freshness insufficient for "
                  "current claims; targeted retry ...")
            retry_queries = _pipe.targeted_queries(topic)
            print("[Research] pass=2 targeted queries="
                  + " | ".join(retry_queries))
            for qi, query in enumerate(retry_queries, start=1):
                attempt(f"arxiv[r{qi}]", self._arxiv, query)
                attempt(f"hn[r{qi}]", self._hn, query)
            pack = _pipe.build_pack(
                topic, collected, max_facts=self.max_facts,
                relevance_threshold=self.relevance_threshold,
                relevance_queries=retry_queries)
            pass_summaries.append(f"pass 2: {len(pack.sources)} source(s), "
                                  f"{pack.research_quality}")
            print(f"[Research] pass=2 sources={len(pack.sources)} "
                  f"quality={pack.research_quality}")
            if not _pipe.current_evidence_sufficient(pack):
                raise ResearchError(
                    "Research error: current-news topic requires recent "
                    "dated evidence, but the research stage could not obtain "
                    "sufficient current sources after retry.\n"
                    f"Research summary: {'; '.join(pass_summaries)}; "
                    "current evidence: insufficient (need 2+ recent dated "
                    "facts from 2+ distinct sources).\n"
                    "Suggested actions: retry later, provide a specific "
                    "source, or change the topic to an evergreen explainer.\n"
                    f"Diagnostics: {'; '.join(diagnostics)}")
        pack.diagnostics = diagnostics
        print(f"[Research] final quality={pack.research_quality}")
        return ResearchPackResult(query=pack.query, sources=pack.sources, pack=pack)
