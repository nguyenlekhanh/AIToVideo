"""Storyboard validation and persistence. JSON is validated before continuing."""
from __future__ import annotations

import json
import os
import re

from subject import SubjectProfile

REQUIRED_FIELDS = ("id", "duration", "image_prompt", "video_prompt", "narration")

# Optional per-scene research grounding (Phase 9). Preserved verbatim when
# present; validated against the research pack only when one is supplied.
GROUNDING_FIELDS = ("research_fact_ids", "source_ids", "grounding")

FACT_MARKERS = re.compile(
    r"\b(latest|current|breakthrough|breakthroughs|announced|announces|"
    r"launched|launches|released|record|first|discover(?:ed|y|ies)|today|"
    r"this year|this month|percent|20\d{2})\b"
    r"|\d+\s*(?:million|billion|thousand)\b|%",
    re.IGNORECASE)
FRESH_MARKERS = re.compile(
    r"\b(latest|current|today|this year|this month|20\d{2})\b", re.IGNORECASE)
YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def validate_storyboard(data: dict, research: dict | None = None) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Storyboard must be a JSON object with a 'scenes' list.")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Storyboard must contain a non-empty 'scenes' list.")
    validated: list[dict] = []
    for i, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene #{i} must be an object.")
        for field in REQUIRED_FIELDS:
            if field == "id":
                continue  # ids are normalized to 1..N below; accept if missing
            if field not in scene:
                raise ValueError(f"Scene #{i} is missing required field: {field!r}")
        try:
            scene_id = int(scene.get("id", i))
        except (TypeError, ValueError):
            scene_id = i
        try:
            duration = int(scene["duration"])
        except (TypeError, ValueError):
            raise ValueError(f"Scene #{i} has non-integer duration: {scene['duration']!r}")
        if duration <= 0:
            raise ValueError(f"Scene #{i} has non-positive duration: {duration}")
        for field in ("image_prompt", "video_prompt"):
            if not isinstance(scene[field], str) or not scene[field].strip():
                raise ValueError(f"Scene #{i} field {field!r} must be a non-empty string.")
        # Narration may be empty (keyframe storyboards default to "") but
        # must be present as a string; audio providers fail clearly on
        # empty narration at generation time.
        if "narration" not in scene or not isinstance(scene["narration"], str):
            raise ValueError(f"Scene #{i} field 'narration' must be a string (may be empty).")
        entry = {
            "id": scene_id,
            "duration": duration,
            "image_prompt": scene["image_prompt"].strip(),
            "video_prompt": scene["video_prompt"].strip(),
            "narration": scene["narration"].strip(),
        }
        # Preserve optional research grounding verbatim (validated below).
        if isinstance(scene.get("research_fact_ids"), list):
            entry["research_fact_ids"] = [str(x) for x in scene["research_fact_ids"]]
        if isinstance(scene.get("source_ids"), list):
            entry["source_ids"] = [str(x) for x in scene["source_ids"]]
        if isinstance(scene.get("grounding"), str) and scene["grounding"].strip():
            entry["grounding"] = scene["grounding"].strip()
        validated.append(entry)
    # normalize ids to 1..N in order
    for i, scene in enumerate(validated, start=1):
        scene["id"] = i
    if research is not None:
        errors = validate_grounding(validated, research)
        if errors:
            raise ValueError("Storyboard grounding errors:\n- " + "\n- ".join(errors))
    return validated


def validate_grounding(scenes: list[dict], research: dict) -> list[str]:
    """Check scene fact/source references against a research metadata dict.

    Returns a list of error strings (empty = grounded). Never invents
    sources to repair a scene; reports every problem found. research=None
    is handled by the caller (no research -> no grounding checks).
    """
    errors: list[str] = []
    facts = {}
    for fact in research.get("facts", []) or []:
        if isinstance(fact, dict) and fact.get("id"):
            facts[str(fact["id"])] = fact
    record_ids = set()
    for record in research.get("source_records", []) or []:
        if isinstance(record, dict) and record.get("id"):
            record_ids.add(str(record["id"]))
    quality = research.get("research_quality", "")
    for i, scene in enumerate(scenes, start=1):
        fids = scene.get("research_fact_ids") or []
        sids = scene.get("source_ids") or []
        grounding = scene.get("grounding", "")
        narration = scene.get("narration", "")
        for fid in fids:
            if fid not in facts:
                errors.append(
                    f"Scene #{i} references nonexistent fact_id {fid!r}.")
        for sid in sids:
            if sid not in record_ids:
                errors.append(
                    f"Scene #{i} references nonexistent source_id {sid!r}.")
        for fid in fids:
            fact = facts.get(fid)
            if fact is None:
                continue
            supported = [s for s in (fact.get("source_ids") or []) if s in record_ids]
            if not supported:
                errors.append(
                    f"Scene #{i} cites fact {fid!r} which has no resolvable source.")
        if fids or grounding == "creative":
            pass
        elif FACT_MARKERS.search(narration):
            errors.append(
                f"Scene #{i} makes factual claims but has no research_fact_ids "
                f"(add grounding or mark grounding: creative).")
        if fids:
            cited_text = " ".join(
                str(facts[f].get("claim", "")) + " "
                + str(facts[f].get("published_at") or "")
                for f in fids if f in facts)
            for year in set(YEAR.findall(narration)):
                if year not in cited_text:
                    errors.append(
                        f"Scene #{i} mentions {year} with no cited fact supporting it.")
            if (FRESH_MARKERS.search(narration)
                    and quality in ("background_only", "insufficient")):
                errors.append(
                    f"Scene #{i} claims latest/current information but research "
                    f"quality is {quality!r}.")
            cited_fresh = {facts[f].get("freshness", "unknown")
                           for f in fids if f in facts}
            if (FRESH_MARKERS.search(narration)
                    and cited_fresh
                    and cited_fresh <= {"unknown", "historical"}):
                errors.append(
                    f"Scene #{i} claims latest/current information but all cited "
                    f"facts are undated/historical.")
    return errors


def _research_facts(research: dict) -> dict:
    """Index research metadata facts by id as ResearchFact objects."""
    from providers.research.base import ResearchFact
    facts = {}
    for fact in research.get("facts", []) or []:
        if isinstance(fact, dict) and fact.get("id"):
            facts[str(fact["id"])] = ResearchFact(
                id=str(fact["id"]), claim=str(fact.get("claim", "")),
                source_ids=[str(s) for s in (fact.get("source_ids") or [])],
                published_at=fact.get("published_at"),
                freshness=str(fact.get("freshness", "unknown")))
    return facts


def assess_scenes(scenes: list[dict], research: dict | None,
                  grounder=None) -> list[dict]:
    """Per-scene semantic assessment: [{index, status, score, reason, fact_ids}].

    Creative scenes are exempt (status "exempt"). Pure assessment: no
    rewriting, no side effects. Powers both validate_semantics and the
    bounded regeneration loop.
    """
    if research is None:
        return []
    if grounder is None:
        from providers.research.semantic import TokenOverlapGrounder
        grounder = TokenOverlapGrounder()
    facts = _research_facts(research)
    assessments = []
    for i, scene in enumerate(scenes, start=1):
        if scene.get("grounding") == "creative":
            assessments.append({"index": i, "status": "exempt", "score": 1.0,
                                "reason": "creative scene exempt",
                                "fact_ids": list(scene.get("research_fact_ids") or [])})
            continue
        fids = [fid for fid in (scene.get("research_fact_ids") or [])
                if fid in facts]
        cited = [facts[fid] for fid in fids]
        result = grounder.ground(scene.get("narration", ""), cited)
        assessments.append({"index": i, "status": result.status,
                            "score": result.score, "reason": result.reason,
                            "fact_ids": fids})
    return assessments


def validate_semantics(scenes: list[dict], research: dict | None,
                       grounder=None) -> tuple[list[str], list[str]]:
    """Layer 2: narration supported by cited facts (not just valid IDs).

    Returns (errors, warnings). Creative scenes are fully exempt (Part G).
    UNSUPPORTED -> error; PARTIAL/UNKNOWN -> warning (regeneration, when
    enabled, consumes these assessments). No research -> ([], []) so
    offline/pack-less flows are unaffected.
    """
    if research is None:
        return [], []
    from providers.research.semantic import (
        STATUS_PARTIAL,
        STATUS_SUPPORTED,
        STATUS_UNKNOWN,
    )
    errors: list[str] = []
    warnings: list[str] = []
    for assessment in assess_scenes(scenes, research, grounder):
        i, status = assessment["index"], assessment["status"]
        if status == STATUS_SUPPORTED or status == "exempt":
            continue
        if status == STATUS_PARTIAL:
            warnings.append(
                f"Scene #{i} partly supported ({assessment['score']}): "
                f"{assessment['reason']}")
        elif status == STATUS_UNKNOWN:
            warnings.append(
                f"Scene #{i} semantic support unknown: {assessment['reason']}")
        else:
            errors.append(
                f"Scene #{i} narration not supported by cited facts "
                f"({assessment['score']}): {assessment['reason']}")
    return errors, warnings


# -- visual-generation gate (deterministic, no model calls) --
#
# Flags image/video prompts that request rendered text, statistics, or
# multi-panel compositions. Research facts belong in narration/metadata,
# never as visible pixels. "no ..." negation clauses are stripped first
# so mandated constraints ("no split screen, no readable text, ...")
# never trip the detector.

_NEGATION_CLAUSE = re.compile(r"\bno\s+[a-z][^.,;]*", re.IGNORECASE)

# (pattern, human-readable reason). Checked against image_prompt and
# video_prompt only -- narration is never inspected here.
VISUAL_TEXT_PATTERNS = (
    (re.compile(r"\"[^\"]{2,}\"|'[^']{3,}'"),
     "quoted text"),
    (re.compile(r"\breadable text\b", re.IGNORECASE),
     "readable text"),
    (re.compile(r"\bheadlines?\b|\bcaptions?\b|\bsubtitles?\b|\blogos?\b",
                re.IGNORECASE),
     "headline/caption/subtitle/logo"),
    (re.compile(r"\binfograph\w*\b", re.IGNORECASE),
     "infographic"),
    (re.compile(r"\bcollage\b|\bsplit.?screen\b|\bside.by.side\b|"
                r"\btwo panels?\b|\bmultiple panels?\b|\bbefore and after\b|"
                r"\bbefore/after\b|\bon one side\b|\bon the other side\b|"
                r"\bmontage\b",
                re.IGNORECASE),
     "multi-panel composition"),
    (re.compile(r"\bcharts?\b|\bgraphs?\b|\bholograms?\b|\bfloating\b|"
                r"\bstatistics?\b",
                re.IGNORECASE),
     "charts/statistics overlay"),
    (re.compile(r"\d{1,3}(,\d{3})+|\b\d+(\.\d+)?\s*%",
                re.IGNORECASE),
     "rendered numbers"),
    (re.compile(r"\b(sign|notice)\b.{0,40}\b(says?|reads?|showing|"
                r"displaying)\b|\b(shows?|displays?|reads?)\s+['\"]",
                re.IGNORECASE),
     "sign/screen showing text"),
    (re.compile(r"\b(screen|monitor|display|newspaper|billboard|document|"
                r"\btv\b|phone|paper|laptop|computer|website|webpage)\s+"
                r"(showing|displaying|with|reads?|says?|containing)\b",
                re.IGNORECASE),
     "prop displaying content"),
    (re.compile(r"\bmonitors?\b|\bscreens?\b|\bcomputers?\b|"
                r"\bnewspapers?\b|\bdocuments?\b|\bcharts?\b|\bgraphs?\b|"
                r"\bmagazines?\b|\bsigns?\b|\bbillboards?\b|\btvs?\b",
                re.IGNORECASE),
     "information-display object"),
)


def _prompt_word_count(text: str) -> int:
    return len(text.split())


def validate_visual_prompts(scenes: list[dict]) -> list[str]:
    """Flag prompts that would render text/numbers or multi-panel layouts.

    Returns a list of error strings (empty = clean). Pure function over
    already-validated scene dicts; narration and grounding metadata are
    never touched.
    """
    errors: list[str] = []
    for i, scene in enumerate(scenes, start=1):
        for field in ("image_prompt", "video_prompt"):
            text = scene.get(field, "")
            if not isinstance(text, str):
                continue
            if _prompt_word_count(text) > 120:
                errors.append(
                    f"Scene #{i} {field} is overloaded "
                    f"({_prompt_word_count(text)} words; target 40-80). "
                    f"Describe only who, where, doing what, emotion, light, "
                    f"camera, style.")
                continue
            cleaned = _NEGATION_CLAUSE.sub("", text)
            for pattern, reason in VISUAL_TEXT_PATTERNS:
                match = pattern.search(cleaned)
                if match:
                    errors.append(
                        f"Scene #{i} {field} requests {reason} "
                        f"({match.group(0).strip()[:60]!r}). Research "
                        f"facts must remain in narration/metadata, not "
                        f"inside the generated image.")
                    break
    return errors


def save_storyboard(scenes: list[dict], path: str, subject=None,
                      research: dict | None = None) -> None:
    """Save scenes plus optional SubjectProfile and research metadata.

    research is a plain dict like {"mode": "web", "sources": [...]{"title",
    "url", "published"}]}. Old files without these keys still load.
    """
    if subject is not None and not isinstance(subject, dict):
        to_dict = getattr(subject, "to_dict", None)
        subject = to_dict() if callable(to_dict) else dict(subject)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"research": research, "subject": subject, "scenes": scenes},
                  f, indent=2, ensure_ascii=False)


def load_storyboard(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return validate_storyboard(json.load(f))


def load_research(path: str) -> dict | None:
    """Restore research metadata from a saved storyboard (None if absent)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    research = data.get("research")
    return research if isinstance(research, dict) else None


def load_subject(path: str) -> SubjectProfile | None:
    """Restore the SubjectProfile from a saved storyboard (None if absent)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return SubjectProfile.from_dict(data.get("subject"))
