"""Research provider interface. main.py only knows these types.

A ResearchProvider collects external information about the user's topic
BEFORE storyboard generation. It never invents facts: on any failure it
raises ResearchError so the pipeline stops instead of generating from
fake/current-looking information.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ResearchError(RuntimeError):
    """Web research failed. Never fall back to fabricated information."""


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str = ""
    published: str | None = None
    metadata: dict = field(default_factory=dict)
    # Phase-3 normalized metadata (empty until the pipeline fills them in).
    id: str = ""
    publisher: str = ""
    published_at: str | None = None
    retrieved_at: str = ""
    source_type: str = "other"

    def to_metadata(self) -> dict:
        """Compact form persisted into storyboard.json (legacy shape)."""
        return {"title": self.title, "url": self.url,
                "published": self.published}

    def to_record(self) -> dict:
        """Full normalized source record (Phase 3). Never fabricates dates."""
        return {"id": self.id, "title": self.title, "url": self.url,
                "publisher": self.publisher, "published_at": self.published_at,
                "retrieved_at": self.retrieved_at,
                "source_type": self.source_type}


@dataclass
class ResearchResult:
    query: str
    sources: list[ResearchSource]

    def render_context(self, max_chars: int = 6000) -> str:
        """Render numbered sources as Ollama context (truncated, labeled)."""
        lines = [f"Web research results for: {self.query}"]
        used = len(lines[0])
        for i, src in enumerate(self.sources, start=1):
            block = f"\n[{i}] {src.title}\nURL: {src.url}\n{src.snippet}".strip()
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        lines.append(
            "\nBase the storyboard on these researched facts where relevant. "
            "Do not invent newer facts beyond what is stated above.")
        return "\n".join(lines)


class ResearchProvider(ABC):
    provider_id: str = "research"

    @abstractmethod
    def research(self, topic: str) -> ResearchResult:
        """Collect sources for the topic. Raises ResearchError on failure."""


# -- Structured Research Pack (Phases 3-8). Deterministic, inspectable. --

FRESH_CURRENT = "current"
FRESH_RECENT = "recent"
FRESH_HISTORICAL = "historical"
FRESH_UNKNOWN = "unknown"

QUALITY_HIGH = "high"
QUALITY_MEDIUM = "medium"
QUALITY_BACKGROUND_ONLY = "background_only"
QUALITY_INSUFFICIENT = "insufficient"


@dataclass
class ResearchFact:
    id: str
    claim: str
    source_ids: list[str] = field(default_factory=list)
    published_at: str | None = None
    freshness: str = FRESH_UNKNOWN
    confidence: float = 0.0

    def to_record(self) -> dict:
        return {"id": self.id, "claim": self.claim,
                "source_ids": list(self.source_ids),
                "published_at": self.published_at,
                "freshness": self.freshness,
                "confidence": self.confidence}


@dataclass
class ResearchConflict:
    id: str
    fact_ids: list[str] = field(default_factory=list)
    description: str = ""

    def to_record(self) -> dict:
        return {"id": self.id, "fact_ids": list(self.fact_ids),
                "description": self.description}


@dataclass
class ResearchPack:
    query: str
    research_quality: str
    retrieved_at: str
    sources: list[ResearchSource] = field(default_factory=list)
    facts: list[ResearchFact] = field(default_factory=list)
    conflicts: list[ResearchConflict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"query": self.query, "research_quality": self.research_quality,
                "retrieved_at": self.retrieved_at,
                "sources": [s.to_record() for s in self.sources],
                "facts": [f.to_record() for f in self.facts],
                "conflicts": [c.to_record() for c in self.conflicts],
                "diagnostics": list(self.diagnostics)}

    def to_context(self, max_chars: int = 6000) -> str:
        """Render the pack for Ollama: facts first, then conflicts, sources."""
        lines = [f"Web research results for: {self.query}",
                 f"Research quality: {self.research_quality} "
                 f"(retrieved {self.retrieved_at})."]
        used = sum(len(line) for line in lines)
        for fact in self.facts:
            block = (f"\n[{fact.id}] {fact.claim}\n"
                     f"Sources: {', '.join(fact.source_ids)}"
                     + (f" | Published: {fact.published_at}"
                        if fact.published_at else " | Date unknown"))
            if used + len(block) > max_chars:
                break
            lines.append(block.strip())
            used += len(block)
        for conflict in self.conflicts:
            block = (f"\n[{conflict.id}] DISAGREEMENT between "
                     f"{', '.join(conflict.fact_ids)}: {conflict.description}")
            if used + len(block) > max_chars:
                break
            lines.append(block.strip())
            used += len(block)
        for i, src in enumerate(self.sources, start=1):
            key = src.id or f"S{i}"
            block = f"\n[{key}] {src.title}\nURL: {src.url}".strip()
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        lines.append(
            "\nBase the storyboard on these researched facts where relevant. "
            "Do not invent newer facts beyond what is stated above.")
        return "\n".join(lines)


@dataclass
class ResearchPackResult(ResearchResult):
    """ResearchResult carrying the structured pack (superset, compatible)."""
    pack: ResearchPack | None = None

    def render_context(self, max_chars: int = 6000) -> str:
        if self.pack is not None:
            return self.pack.to_context(max_chars)
        return super().render_context(max_chars)
