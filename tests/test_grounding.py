"""Unit tests: source metadata, freshness, facts, grounding, noise (no network).

Phase 13 acceptance fixture: one recent official source, one recent news
source, one old encyclopedic source, one irrelevant source, one duplicate.
"""
from __future__ import annotations

import datetime as _dt
import unittest

import storyboard as sb
from providers.research import pipeline as pipe
from providers.research.base import ResearchSource


def days_ago(n: int) -> str:
    return (_dt.date.today() - _dt.timedelta(days=n)).isoformat()


def src(title, url, snippet, publisher="", published_at=None, source_type="other"):
    return ResearchSource(title=title, url=url, snippet=snippet,
                          publisher=publisher, published_at=published_at,
                          source_type=source_type)


def acceptance_sources():
    official = src(
        "Figure AI Unveils Figure 02 Humanoid Robot",
        "https://figure.ai/news/figure-02",
        "Figure AI unveiled its Figure 02 humanoid robot in August 2024. "
        "The robot stands 168 centimeters tall and weighs 70 kilograms. "
        "Figure 02 uses onboard artificial intelligence for autonomous tasks.",
        publisher="Figure AI", published_at=days_ago(10), source_type="official")
    news = src(
        "Boston Dynamics Atlas Shows New Backflip Routine",
        "https://example-news.com/atlas-backflip",
        "Boston Dynamics demonstrated a new Atlas backflip routine in late 2025. "
        "The hydraulic humanoid completed three consecutive backflips on video. "
        "Engineers say the routine required months of reinforcement learning.",
        publisher="Example News", published_at=days_ago(20), source_type="news")
    encyclopedic = src(
        "Humanoid robot",
        "https://en.wikipedia.org/wiki/Humanoid_robot",
        "A humanoid robot is a robot resembling the human body in shape. "
        "The design may be aimed at functional purposes such as interacting "
        "with human tools and environments and working alongside humans.",
        publisher="Wikipedia", published_at=None, source_type="encyclopedic")
    irrelevant = src(
        "China-United States relations",
        "https://en.wikipedia.org/wiki/China-United_States_relations",
        "China-United States relations involve trade, diplomacy and tariffs "
        "between the two countries over several decades of negotiation.",
        publisher="Wikipedia", published_at=None, source_type="encyclopedic")
    duplicate = src(
        "Figure 02 Humanoid Robot Debuts",
        "https://example-tech.com/figure-02-debut",
        "Figure AI unveiled its Figure 02 humanoid robot in August 2024. "
        "The new humanoid features advanced onboard artificial intelligence.",
        publisher="Example Tech", published_at=days_ago(12), source_type="news")
    return [official, news, encyclopedic, irrelevant, duplicate]


QUERY = "Humanoid robots latest breakthroughs innovations"


class MetadataTest(unittest.TestCase):
    def test_record_preserves_all_fields(self):
        source = src("T", "https://e.com/x", "S", publisher="P",
                     published_at="2026-01-15", source_type="news")
        source.id, source.retrieved_at = "src_001", "2026-09-30T00:00:00+00:00"
        record = source.to_record()
        self.assertEqual(record["url"], "https://e.com/x")
        self.assertEqual(record["title"], "T")
        self.assertEqual(record["publisher"], "P")
        self.assertEqual(record["published_at"], "2026-01-15")
        self.assertEqual(record["retrieved_at"], "2026-09-30T00:00:00+00:00")
        self.assertEqual(record["source_type"], "news")

    def test_missing_date_stays_null(self):
        source = src("T", "https://e.com/x", "S")
        self.assertIsNone(source.to_record()["published_at"])

    def test_retrieved_at_populated_by_pipeline(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        for record in [s.to_record() for s in pack.sources]:
            self.assertTrue(record["retrieved_at"])
            self.assertTrue(record["id"].startswith("src_"))


class FreshnessTest(unittest.TestCase):
    def test_classification_matrix(self):
        self.assertEqual(pipe.classify_freshness(days_ago(5)), "current")
        self.assertEqual(pipe.classify_freshness(days_ago(100)), "recent")
        self.assertEqual(pipe.classify_freshness(days_ago(400)), "historical")
        self.assertEqual(pipe.classify_freshness(None), "unknown")
        self.assertEqual(pipe.classify_freshness("not-a-date"), "unknown")

    def test_undated_encyclopedic_never_current(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        wiki = [s for s in pack.sources if s.source_type == "encyclopedic"]
        self.assertTrue(wiki)
        for source in wiki:
            self.assertNotIn(
                pipe.classify_freshness(source.published_at, source.source_type),
                ("current", "recent"))

    def test_latest_query_prefers_recent(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        fresh = [f for f in pack.facts
                 if f.freshness in ("current", "recent")]
        self.assertTrue(fresh, "latest query must yield recent-dated facts")
        self.assertIn(pack.research_quality, ("high", "medium"))


class FactTest(unittest.TestCase):
    def test_facts_have_sources(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        self.assertTrue(pack.facts)
        ids = {s.id for s in pack.sources}
        for fact in pack.facts:
            self.assertTrue(fact.source_ids)
            for sid in fact.source_ids:
                self.assertIn(sid, ids, "never invent source IDs")

    def test_duplicates_merge(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        figure_facts = [f for f in pack.facts if "Figure 02" in f.claim]
        self.assertEqual(len(figure_facts), 1)
        self.assertGreaterEqual(len(figure_facts[0].source_ids), 2)

    def test_conflict_detected(self):
        from providers.research.base import ResearchFact
        facts = [
            ResearchFact(id="fact_001",
                         claim="The Atlas humanoid uses 24 hydraulic joints for movement.",
                         source_ids=["src_001"]),
            ResearchFact(id="fact_002",
                         claim="The Atlas humanoid uses 31 hydraulic joints for movement.",
                         source_ids=["src_002"]),
        ]
        conflicts = pipe.detect_conflicts(facts)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(set(conflicts[0].fact_ids), {"fact_001", "fact_002"})

    def test_no_conflict_without_numbers(self):
        from providers.research.base import ResearchFact
        facts = [
            ResearchFact(id="fact_001", claim="Atlas can do backflips on video.",
                         source_ids=["src_001"]),
            ResearchFact(id="fact_002", claim="Atlas performs backflips on video.",
                         source_ids=["src_002"]),
        ]
        self.assertEqual(pipe.detect_conflicts(facts), [])


class NoiseTest(unittest.TestCase):
    def test_irrelevant_source_filtered(self):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        titles = [s.title for s in pack.sources]
        self.assertNotIn("China-United States relations", titles)
        claims = " ".join(f.claim for f in pack.facts)
        self.assertNotIn("China", claims)
        self.assertNotIn("tariffs", claims)


class GroundingValidationTest(unittest.TestCase):
    def _meta(self, **overrides):
        pack = pipe.build_pack(QUERY, acceptance_sources())
        meta = {"mode": "web",
                "sources": [s.to_metadata() for s in pack.sources],
                "query": pack.query, "research_quality": pack.research_quality,
                "retrieved_at": pack.retrieved_at,
                "facts": [f.to_record() for f in pack.facts],
                "conflicts": [c.to_record() for c in pack.conflicts],
                "source_records": [s.to_record() for s in pack.sources]}
        meta.update(overrides)
        return pack, meta

    def _scene(self, **overrides):
        base = {"id": 1, "duration": 5, "image_prompt": "robot lab",
                "video_prompt": "pan", "narration": "A robot stands in a lab."}
        base.update(overrides)
        return base

    def test_valid_grounded_scene(self):
        pack, meta = self._meta()
        fid = pack.facts[0].id
        sid = pack.facts[0].source_ids[0]
        scenes = sb.validate_storyboard(
            {"scenes": [self._scene(research_fact_ids=[fid], source_ids=[sid])]},
            meta)
        self.assertEqual(scenes[0]["research_fact_ids"], [fid])

    def test_unknown_fact_id_rejected(self):
        _, meta = self._meta()
        with self.assertRaises(ValueError) as ctx:
            sb.validate_storyboard(
                {"scenes": [self._scene(research_fact_ids=["fact_999"])]}, meta)
        self.assertIn("fact_999", str(ctx.exception))

    def test_unknown_source_id_rejected(self):
        pack, meta = self._meta()
        fid = pack.facts[0].id
        with self.assertRaises(ValueError) as ctx:
            sb.validate_storyboard(
                {"scenes": [self._scene(research_fact_ids=[fid],
                                        source_ids=["src_999"])]}, meta)
        self.assertIn("src_999", str(ctx.exception))

    def test_factual_scene_without_grounding_rejected(self):
        _, meta = self._meta()
        with self.assertRaises(ValueError) as ctx:
            sb.validate_storyboard({"scenes": [self._scene(
                narration="The latest breakthrough was announced in 2026.")]}, meta)
        self.assertIn("no research_fact_ids", str(ctx.exception))

    def test_creative_scene_exempt(self):
        _, meta = self._meta()
        scenes = sb.validate_storyboard({"scenes": [self._scene(
            narration="The latest breakthrough was announced in 2026.",
            grounding="creative")]}, meta)
        self.assertEqual(scenes[0]["grounding"], "creative")

    def test_plain_scene_without_markers_passes(self):
        _, meta = self._meta()
        scenes = sb.validate_storyboard(
            {"scenes": [self._scene(narration="A robot stands quietly.")]}, meta)
        self.assertNotIn("research_fact_ids", scenes[0])

    def test_fabricated_year_rejected(self):
        pack, meta = self._meta()
        fid = next(f.id for f in pack.facts)
        with self.assertRaises(ValueError) as ctx:
            sb.validate_storyboard({"scenes": [self._scene(
                narration="In 2031 the robot ruled the world.",
                research_fact_ids=[fid])]}, meta)
        self.assertIn("2031", str(ctx.exception))

    def test_no_research_compat(self):
        scenes = sb.validate_storyboard({"scenes": [self._scene(
            narration="The latest breakthrough was announced in 2026.")]}, None)
        self.assertEqual(len(scenes), 1)


class AcceptanceTraceTest(unittest.TestCase):
    def test_full_trace_scene_to_url(self):
        """scene -> fact -> source -> URL -> published_at (Phase 13)."""
        pack = pipe.build_pack(QUERY, acceptance_sources())
        recent_official = [f for f in pack.facts
                           if f.freshness in ("current", "recent")
                           and any(s.publisher in ("Figure AI", "Example Tech",
                                                  "Example News")
                                   for s in pack.sources if s.id in f.source_ids)]
        self.assertTrue(recent_official, "recent non-encyclopedic facts required")
        fact = recent_official[0]
        by_id = {s.id: s for s in pack.sources}
        for sid in fact.source_ids:
            self.assertIn(sid, by_id)
            src = by_id[sid]
            self.assertTrue(src.url.startswith("http"))
            if src.source_type in ("official", "news"):
                self.assertTrue(src.published_at,
                                f"dated source {sid} must keep published_at")
        # A scene citing this fact validates against the persisted metadata.
        meta = {"mode": "web", "sources": [],
                "facts": [f.to_record() for f in pack.facts],
                "conflicts": [], "source_records": [s.to_record() for s in pack.sources]}
        sid = fact.source_ids[0]
        scenes = sb.validate_storyboard(
            {"scenes": [{"id": 1, "duration": 5, "image_prompt": "p",
                         "video_prompt": "v",
                         "narration": f"Robots advance: {fact.claim}",
                         "research_fact_ids": [fact.id], "source_ids": [sid]}]},
            meta)
        self.assertEqual(scenes[0]["research_fact_ids"], [fact.id])


class SemanticLayerTest(unittest.TestCase):
    def _meta_for(self, claim, published_at=None, freshness="recent"):
        return {"mode": "web", "sources": [],
                "facts": [{"id": "fact_001", "claim": claim,
                           "source_ids": ["src_001"],
                           "published_at": published_at,
                           "freshness": freshness, "confidence": 0.8}],
                "conflicts": [],
                "source_records": [{"id": "src_001", "title": "T",
                                    "url": "https://example.com/x",
                                    "publisher": "Example",
                                    "published_at": published_at,
                                    "retrieved_at": "2026-01-01T00:00:00+00:00",
                                    "source_type": "news"}]}

    def _scene(self, narration, fids=("fact_001",), grounding="researched"):
        scene = {"id": 1, "duration": 5, "image_prompt": "p",
                 "video_prompt": "v", "narration": narration,
                 "research_fact_ids": list(fids),
                 "source_ids": ["src_001"], "grounding": grounding}
        return [scene]

    def test_supported_scene_passes(self):
        meta = self._meta_for("NVIDIA introduced the DGX A100 system in 2020.",
                              "2020-05-14")
        errors, warnings = sb.validate_semantics(
            self._scene("NVIDIA introduced the DGX A100 system."), meta)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_unrelated_narration_fails(self):
        meta = self._meta_for("NVIDIA introduced the DGX A100 system in 2020.",
                              "2020-05-14")
        errors, _ = sb.validate_semantics(
            self._scene("NVIDIA's AI hardware has transformed humanoid robotics."),
            meta)
        self.assertTrue(errors)
        self.assertIn("not supported", errors[0])

    def test_partial_warns_only(self):
        meta = self._meta_for("Atlas can jump.")
        errors, warnings = sb.validate_semantics(
            self._scene("Atlas can jump and fly over tall buildings."), meta)
        self.assertEqual(errors, [])
        self.assertTrue(warnings)

    def test_creative_exempt(self):
        meta = self._meta_for("NVIDIA introduced the DGX A100 system in 2020.",
                              "2020-05-14")
        errors, warnings = sb.validate_semantics(
            self._scene("A robot dreams of electric sheep in 2026.",
                        fids=(), grounding="creative"), meta)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_no_research_noop(self):
        errors, warnings = sb.validate_semantics(
            self._scene("Anything at all."), None)
        self.assertEqual((errors, warnings), ([], []))

    def test_stub_grounder_injected(self):
        from providers.research.semantic import GroundingResult
        meta = self._meta_for("Atlas can jump.")

        class AlwaysPartial:
            def ground(self, narration, facts):
                return GroundingResult("partial", 0.5,
                                       [f.id for f in facts], "stub")
        errors, warnings = sb.validate_semantics(
            self._scene("Atlas can jump."), meta, grounder=AlwaysPartial())
        self.assertEqual(errors, [])
        self.assertTrue(warnings)

    def test_research_metadata_preserved(self):
        import copy
        meta = self._meta_for("Atlas can jump.")
        snapshot = copy.deepcopy(meta)
        sb.validate_semantics(self._scene("Atlas can jump."), meta)
        sb.validate_storyboard(
            {"scenes": [{"id": 1, "duration": 5, "image_prompt": "p",
                         "video_prompt": "v", "narration": "Atlas can jump.",
                         "research_fact_ids": ["fact_001"],
                         "source_ids": ["src_001"]}]}, meta)
        self.assertEqual(meta, snapshot)


if __name__ == "__main__":
    unittest.main()
