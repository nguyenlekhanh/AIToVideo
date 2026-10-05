"""One-off: rewrite job-market image/video prompts (simple literal rule).

Loads the existing board, validates STRUCTURE only (the file's research
block is stale from an earlier fetch; groundingwan/narration are
preserved byte-for-byte, never revalidated or rewritten), runs the
standard bounded ensure_clean_visuals loop, saves back with the original
research + subject intact.
"""
import json
import sys

sys.path.insert(0, ".")

import main
import ollama as ol
import storyboard as sb
from subject import SubjectProfile

PATH = "projects/job_market_research/storyboard.json"
MODEL = "qwen3:8b"


def main_regen():
    with open(PATH, encoding="utf-8") as f:
        raw = json.load(f)
    research = raw.get("research")
    scenes = sb.validate_storyboard(raw, None)
    print(f"loaded {len(scenes)} scenes (structure ok, "
          f"grounding untouched)")

    def rewrite_fn(scene, reason):
        return ol.rewrite_scene_visuals(
            scene, reason, model=MODEL,
            base_url="http://127.0.0.1:11434", timeout=180)

    scenes, count = main.ensure_clean_visuals(scenes, rewrite_fn)
    print(f"visual rewrites: {count}")
    before = {(s["id"]): (s["narration"], s.get("research_fact_ids"),
                          s.get("source_ids"), s.get("grounding"))
              for s in raw["scenes"]}
    for s in scenes:
        old = before[s["id"]]
        new = (s["narration"], s.get("research_fact_ids"),
               s.get("source_ids"), s.get("grounding"))
        assert old == new, f"scene {s['id']} narration/grounding changed!"
    print("narration + grounding byte-identical for all scenes")
    subject = SubjectProfile.from_dict(raw.get("subject"))
    sb.save_storyboard(scenes, PATH, subject, research)
    print(f"saved {len(scenes)} scenes -> {PATH}")
    remaining = sb.validate_visual_prompts(scenes)
    print(f"remaining visual errors: {len(remaining)}")
    for error in remaining:
        print(" -", error[:150])
    for s in scenes:
        print(f"scene {s['id']}: {len(s['image_prompt'].split())} words")


if __name__ == "__main__":
    main_regen()
