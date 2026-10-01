"""Deterministic semantic grounding: narration vs cited facts.

No models, no downloads, no network: token/entity/number overlap heuristics
with conservative thresholds. Exposes the SemanticGrounder interface so a
model-based grounder can be plugged in later without touching validation.

Statuses: supported (entailed) / partial (some support + extra claims) /
unsupported (contradicts or goes beyond) / unknown (cannot determine).
When uncertain, strong factual claims are NOT marked supported.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .base import ResearchFact
from .pipeline import _NUMBER, content_tokens

STATUS_SUPPORTED = "supported"
STATUS_PARTIAL = "partial"
STATUS_UNSUPPORTED = "unsupported"
STATUS_UNKNOWN = "unknown"

COVER_SUPPORT = 0.70
COVER_PARTIAL = 0.35

# Superlative/absolutist words: a narration introducing one absent from the
# cited facts is making a stronger claim than its support.
STRONG_WORDS = frozenset(
    "latest first largest fastest slowest biggest smallest best worst most "
    "only breakthrough record all every never always fastest unique "
    "unprecedented revolutionary".split())

_MULTI_PROPER = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b")


def _norm_number(token: str) -> str:
    return re.sub(r"[^0-9]", "", token)


@dataclass
class GroundingResult:
    status: str
    score: float
    fact_ids: list[str] = field(default_factory=list)
    reason: str = ""


class SemanticGrounder(ABC):
    """Pluggable semantic check: narration supported by cited facts?"""

    @abstractmethod
    def ground(self, narration: str,
               facts: list[ResearchFact]) -> GroundingResult:
        """Return SUPPORTED/PARTIAL/UNSUPPORTED/UNKNOWN (deterministic)."""


class TokenOverlapGrounder(SemanticGrounder):
    """Conservative token/entity/number overlap grounder (no model)."""

    def ground(self, narration: str,
               facts: list[ResearchFact]) -> GroundingResult:
        narration = (narration or "").strip()
        usable = [f for f in facts if f is not None and (f.claim or "").strip()]
        if not usable:
            return GroundingResult(STATUS_UNKNOWN, 0.0, [],
                                   "no cited facts to check against")
        fact_ids = [f.id for f in usable]
        support_text = " ".join(f.claim for f in usable)
        support_tokens = set()
        for fact in usable:
            support_tokens |= content_tokens(fact.claim)
        narr_tokens = content_tokens(narration)
        if not narr_tokens:
            return GroundingResult(STATUS_UNKNOWN, 0.0, fact_ids,
                                   "empty narration")

        support_numbers = {_norm_number(n) for n in _NUMBER.findall(support_text)}
        support_numbers.discard("")
        new_numbers = sorted({n for n in _NUMBER.findall(narration)
                              if _norm_number(n) not in support_numbers})
        if new_numbers:
            return GroundingResult(
                STATUS_UNSUPPORTED, 0.0, fact_ids,
                f"narration introduces numbers/dates not in cited facts: "
                f"{', '.join(new_numbers)}")

        lowered_support = support_text.lower()
        new_entities = sorted({m for m in _MULTI_PROPER.findall(narration)
                               if m.lower() not in lowered_support})
        if new_entities:
            return GroundingResult(
                STATUS_UNSUPPORTED, 0.0, fact_ids,
                f"narration names entities absent from cited facts: "
                f"{', '.join(new_entities)}")

        new_strong = sorted({w for w in narr_tokens
                             if w in STRONG_WORDS and w not in support_tokens})
        coverage = (len(narr_tokens & support_tokens) / len(narr_tokens)
                    if narr_tokens else 0.0)
        score = round(coverage, 2)
        uncovered = sorted(narr_tokens - support_tokens)
        if new_strong:
            if coverage < 0.5:
                return GroundingResult(
                    STATUS_UNSUPPORTED, score, fact_ids,
                    f"unsupported superlative/absolutist terms "
                    f"{', '.join(new_strong)} with low overlap "
                    f"({', '.join(uncovered[:6])} not in cited facts)")
            return GroundingResult(
                STATUS_PARTIAL, score, fact_ids,
                f"superlative/absolutist terms beyond cited facts: "
                f"{', '.join(new_strong)}")
        if coverage >= COVER_SUPPORT:
            return GroundingResult(STATUS_SUPPORTED, score, fact_ids,
                                   "narration covered by cited facts")
        if coverage >= COVER_PARTIAL:
            return GroundingResult(
                STATUS_PARTIAL, score, fact_ids,
                f"partly covered; unsupported terms: "
                f"{', '.join(uncovered[:8])}")
        return GroundingResult(
            STATUS_UNSUPPORTED, score, fact_ids,
            f"narration goes beyond cited facts: "
            f"{', '.join(uncovered[:8])}")
