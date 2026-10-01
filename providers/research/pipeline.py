"""Deterministic research pipeline: normalize -> filter -> facts ->
dedupe -> conflicts -> quality -> ResearchPack.

No LLMs, no network here: pure functions over collected sources, so every
step is unit-testable with fixtures. Never invents dates, IDs, or claims.
"""
from __future__ import annotations

import datetime as _dt
import re

from .base import (
    FRESH_CURRENT,
    FRESH_HISTORICAL,
    FRESH_RECENT,
    FRESH_UNKNOWN,
    QUALITY_BACKGROUND_ONLY,
    QUALITY_HIGH,
    QUALITY_INSUFFICIENT,
    QUALITY_MEDIUM,
    ResearchConflict,
    ResearchFact,
    ResearchPack,
    ResearchSource,
)

CURRENT_DAYS = 31
RECENT_DAYS = 365
DEDUP_JACCARD = 0.55
CONFLICT_SHARED_TOKENS = 3

FRESHNESS_MARKERS = re.compile(
    r"\b(latest|recent|recently|new|newest|newly announced|breakthrough|"
    r"breakthroughs|announced|announces|launched|launches|recently launched|"
    r"released|current|today|this year|this month|this week|2026|20\d{2})\b",
    re.IGNORECASE)

TEMPORAL_STRIP = re.compile(
    r"\b(latest|current|recent|recently|newest|new|today|this week|"
    r"this month|this year|newly announced|recently launched|breakthrough|"
    r"breakthroughs|announced|announces|launched|launches|released|"
    r"20\d{2})\b",
    re.IGNORECASE)

# Authority weights for deterministic source scoring. Ordered so that,
# at comparable relevance: recent authoritative > recent generic >
# old authoritative background > undated background.
TYPE_WEIGHTS = {"official": 0.9, "research": 0.8, "news": 0.7,
                "technical": 0.5, "encyclopedic": 0.3, "other": 0.2}
RECENCY_WEIGHTS = {FRESH_CURRENT: 3.0, FRESH_RECENT: 2.0,
                   FRESH_HISTORICAL: 1.0, FRESH_UNKNOWN: 0.0}

_CLAIM_SPLIT = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,.]*")
_STOP = frozenset(
    "the a an of and or to in on for with by from as at is are was were be "
    "been it its this that these those their there here which who whom what "
    "when where how why not no nor can will just than then than so such only "
    "into over under about".split())


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _stem(word: str) -> str:
    """Crude English stemming so 'robots' matches 'robotics'/'robot'."""
    if len(word) > 4 and word.endswith("ics"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def content_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())
            if w not in _STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def wants_fresh(topic: str) -> bool:
    """True when the topic asks for latest/current information."""
    return FRESHNESS_MARKERS.search(topic or "") is not None


def strip_temporal_markers(topic: str) -> str:
    """Remove temporal-intent words/years, leaving the stable base topic."""
    base = TEMPORAL_STRIP.sub(" ", topic or "")
    return re.sub(r"\s+", " ", base).strip(" ,:-")


def expand_queries(topic: str, max_expansions: int = 2) -> list[str]:
    """Query variants for temporal intent: [original] + targeted searches.

    Generic (not topic-specific): the base topic plus year/announcement
    variants so dated fetchers get several chances at recent sources.
    Non-temporal topics return just the original query.
    """
    original = (topic or "").strip()
    if not original or not wants_fresh(original):
        return [original] if original else []
    year = str(_dt.date.today().year)
    base = strip_temporal_markers(original) or original
    variants = [f"{base} {year}", f"{base} announcement {year}"]
    out = [original]
    for variant in variants:
        variant = re.sub(r"\s+", " ", variant).strip()
        if variant and variant not in out:
            out.append(variant)
        if len(out) - 1 >= max_expansions:
            break
    return out


def source_score(source: ResearchSource, query: str) -> float:
    """Deterministic score: recency dominates, then authority, then relevance.

    Guarantees: recent authoritative > recent generic > old authoritative
    background > undated background (at comparable relevance). Applied
    AFTER explicit relevance filtering, never instead of it.
    """
    freshness = classify_freshness(source.published_at, source.source_type)
    return (RECENCY_WEIGHTS[freshness]
            + TYPE_WEIGHTS.get(source.source_type, 0.2)
            + (0.2 if source.published_at else 0.0)
            + 0.3 * relevance_score(query, source.title, source.snippet))


def parse_date(value: str | None):
    """Parse YYYY-MM-DD (or YYYY-MM / YYYY) to a date. None when unknown."""
    if not value or not isinstance(value, str):
        return None
    m = re.match(r"^\s*(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", value)
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1))
    except ValueError:
        return None


def classify_freshness(published_at: str | None, source_type: str = "",
                       now: _dt.date | None = None) -> str:
    """Classify a source/fact. Undated content is NEVER current/recent."""
    day = parse_date(published_at)
    if day is None:
        return FRESH_UNKNOWN
    today = now or _dt.date.today()
    age = (today - day).days
    if age < 0:
        return FRESH_UNKNOWN  # future dates are data errors, not facts
    if age <= CURRENT_DAYS:
        return FRESH_CURRENT
    if age <= RECENT_DAYS:
        return FRESH_RECENT
    return FRESH_HISTORICAL


def relevance_score(query: str, title: str, snippet: str) -> float:
    """Fraction of query content tokens present in title+snippet."""
    qtokens = content_tokens(query)
    if not qtokens:
        return 0.0
    hay = content_tokens(f"{title} {snippet}")
    return len(qtokens & hay) / len(qtokens)


def sort_sources_by_score(sources: list[ResearchSource],
                          query: str) -> list[ResearchSource]:
    """Stable best-first ordering by deterministic source score."""
    return sorted(sources, key=lambda src: source_score(src, query),
                  reverse=True)


def normalize_sources(query: str, sources: list[ResearchSource],
                      retrieved_at: str, relevance_threshold: float = 0.15,
                      ) -> list[ResearchSource]:
    """Assign ids/timestamps, drop retrieval noise, rank by source score.

    Relevance filtering runs FIRST (noise never survives); survivors are
    stably sorted by deterministic score so recent/authoritative sources
    come first and ids stay dense within the run.
    """
    kept: list[ResearchSource] = []
    for src in sources:
        score = relevance_score(query, src.title, src.snippet)
        if score < relevance_threshold:
            continue
        src.retrieved_at = retrieved_at
        kept.append(src)
    kept = sort_sources_by_score(kept, query)
    for i, src in enumerate(kept, start=1):
        src.id = f"src_{i:03d}"
    return kept


def split_claims(text: str, min_words: int = 6) -> list[str]:
    out = []
    for sentence in _CLAIM_SPLIT.split(text or ""):
        sentence = " ".join(sentence.split())
        if len(sentence.split()) >= min_words:
            out.append(sentence)
    return out


def extract_facts(sources: list[ResearchSource], query: str,
                  max_facts: int = 8, per_source: int = 2) -> list[ResearchFact]:
    """Extract top sentences per source as fact claims with source support."""
    qtokens = content_tokens(query)
    facts: list[ResearchFact] = []
    n = 0
    for src in sources:
        scored = []
        for claim in split_claims(src.snippet):
            overlap = len(content_tokens(claim) & qtokens)
            if overlap == 0:
                continue
            scored.append((overlap, claim))
        scored.sort(key=lambda item: -item[0])
        for overlap, claim in scored[:per_source]:
            n += 1
            facts.append(ResearchFact(
                id=f"fact_{n:03d}", claim=claim, source_ids=[src.id],
                published_at=src.published_at,
                freshness=classify_freshness(src.published_at, src.source_type),
                confidence=round(min(0.95, 0.5 + 0.1 * overlap), 2)))
            if len(facts) >= max_facts:
                return facts
    return facts


def dedupe_facts(facts: list[ResearchFact],
                 threshold: float = DEDUP_JACCARD) -> list[ResearchFact]:
    """Merge equivalent claims (token Jaccard); union source_ids."""
    merged: list[ResearchFact] = []
    for fact in facts:
        tokens = content_tokens(fact.claim)
        placed = False
        for keep in merged:
            if jaccard(tokens, content_tokens(keep.claim)) >= threshold:
                for sid in fact.source_ids:
                    if sid not in keep.source_ids:
                        keep.source_ids.append(sid)
                keep.source_ids.sort()
                keep.confidence = max(keep.confidence, fact.confidence)
                placed = True
                break
        if not placed:
            merged.append(fact)
    for i, fact in enumerate(merged, start=1):
        fact.id = f"fact_{i:03d}"
    return merged


def detect_conflicts(facts: list[ResearchFact]) -> list[ResearchConflict]:
    """Flag fact pairs sharing topic tokens but disagreeing on numbers/dates."""
    conflicts: list[ResearchConflict] = []
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            a, b = facts[i], facts[j]
            shared = content_tokens(a.claim) & content_tokens(b.claim)
            if len(shared) < CONFLICT_SHARED_TOKENS:
                continue
            nums_a = set(_NUMBER.findall(a.claim))
            nums_b = set(_NUMBER.findall(b.claim))
            if nums_a and nums_b and nums_a.isdisjoint(nums_b):
                conflicts.append(ResearchConflict(
                    id=f"conflict_{len(conflicts) + 1:03d}",
                    fact_ids=[a.id, b.id],
                    description=(f"Sources disagree: '{a.claim[:120]}...' "
                                 f"vs '{b.claim[:120]}...'")))
    return conflicts


def assess_quality(facts: list[ResearchFact], topic: str) -> str:
    """Quality gate. Fresh queries without recent dated facts are background_only."""
    if not facts:
        return QUALITY_INSUFFICIENT
    recent = [f for f in facts if f.freshness in (FRESH_CURRENT, FRESH_RECENT)]
    if wants_fresh(topic) and not recent:
        return QUALITY_BACKGROUND_ONLY
    distinct_sources = {s for f in facts for s in f.source_ids}
    if len(recent) >= 2 and len(distinct_sources) >= 2:
        return QUALITY_HIGH
    if recent:
        return QUALITY_MEDIUM
    non_encyclopedic = [f for f in facts if f.freshness != FRESH_UNKNOWN]
    if non_encyclopedic:
        return QUALITY_MEDIUM
    return QUALITY_BACKGROUND_ONLY


def build_pack(query: str, sources: list[ResearchSource],
               max_facts: int = 8, relevance_threshold: float = 0.15,
               retrieved_at: str = "") -> ResearchPack:
    """Full deterministic pipeline from collected sources to ResearchPack."""
    retrieved_at = retrieved_at or utcnow_iso()
    kept = normalize_sources(query, sources, retrieved_at, relevance_threshold)
    facts = dedupe_facts(extract_facts(kept, query, max_facts))
    conflicts = detect_conflicts(facts)
    quality = assess_quality(facts, query)
    return ResearchPack(query=query.strip(), research_quality=quality,
                        retrieved_at=retrieved_at, sources=kept,
                        facts=facts, conflicts=conflicts)
