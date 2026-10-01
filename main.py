"""CLI pipeline. Model-agnostic: only providers + generic requests live here.

Pipeline: Ollama storyboard -> ImageProvider -> VideoProvider ->
AudioProvider -> FFmpeg final.mp4. No node IDs, model filenames, latent
logic, or sampler settings appear in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ffmpeg as ff
import ollama as ol
import storyboard as sb
import state as st
from subject import SubjectProfile, compose_scene_prompt
from providers.audio.base import AudioRequest
from providers.comfy import ComfyClient
from providers.dims import resolve_dimensions
from providers.errors import ProviderError, UnknownModelError
from providers.image.base import ImageRequest
from providers.registry import create_provider, lookup
from providers.video.base import VideoRequest

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None  # type: ignore
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _coerce(value)
    return root


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def slugify(name: str) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return name or "project"


STOP_AFTER_ALIASES = {"images": "image", "videos": "video"}


def normalize_stop_after(value: str) -> str:
    """Canonical stage gate: image(s)/video(s) accept singular and plural."""
    return STOP_AFTER_ALIASES.get(value, value)


def validate_cli_combination(args) -> str | None:
    """Check flag combinations that argparse cannot express.

    Returns an error message, or None when the combination is valid.
    Pure function (no I/O) so unit tests can cover every combination.
    """
    if args.prompt is None and not args.resume:
        return "the prompt is required unless --resume is used"
    if args.resume and not args.project:
        return "--resume requires --project"
    if args.stage is not None and not args.resume:
        return "--stage requires --resume"
    if args.scene is not None and not args.resume:
        return "--scene requires --resume"
    if args.scene is not None and args.scene < 1:
        return "--scene must be a positive 1-based scene number"
    if args.stage is not None and normalize_stop_after(args.stop_after) != "all":
        return "--stage and --stop-after are mutually exclusive"
    return None


def resolve_scene_duration(scene: dict, default_duration: float) -> float:
    """Storyboard is the source of truth for per-scene duration.

    Returns scene["duration"] when present and a positive number, else the
    configured default (legacy/invalid storyboards). Never invents values.
    """
    try:
        duration = float(scene.get("duration", default_duration))
    except (TypeError, ValueError):
        return default_duration
    if duration <= 0:
        return default_duration
    return duration


def ensure_grounded_scenes(scenes, research_meta, rewrite_fn, max_rounds=2):
    """Bounded single-scene regeneration for semantically unsupported scenes.

    scenes: validated scene list (mutated in place on rewrite). rewrite_fn
    takes (scene, cited_fact_records) and returns an updated scene with a
    new narration (same IDs, no new facts/sources). Returns
    (scenes, rewrite_count, warnings). Raises ValueError when scenes are
    still unsupported after max_rounds. Structural errors are the caller's
    responsibility (validate_storyboard runs before this).
    """
    warnings_all: list = []
    rewrites = 0
    facts = {}
    if isinstance(research_meta, dict):
        for fact in research_meta.get("facts", []) or []:
            if isinstance(fact, dict) and fact.get("id"):
                facts[str(fact["id"])] = fact
    for _ in range(max_rounds + 1):
        errors, warnings = sb.validate_semantics(scenes, research_meta)
        warnings_all = warnings  # latest round only; fixed scenes drop old warnings
        failing = [a for a in sb.assess_scenes(scenes, research_meta)
                   if a["status"] == "unsupported"]
        if not failing:
            return scenes, rewrites, warnings_all
        if rewrites >= max_rounds:
            raise ValueError("Storyboard semantic grounding errors:\n- "
                             + "\n- ".join(errors))
        for assessment in failing:
            scene = scenes[assessment["index"] - 1]
            cited = [facts[fid] for fid in assessment["fact_ids"] if fid in facts]
            scenes[assessment["index"] - 1] = rewrite_fn(
                dict(scene), cited, assessment["reason"])
            rewrites += 1
    raise AssertionError("unreachable")  # loop always returns or raises above


class StageInterrupted(Exception):
    """Ctrl+C during a stage loop. Carries a resume-ready message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def scene_path(directory: str, scene_id: int, ext: str) -> str:
    """Stable per-scene output name, e.g. scene_003.png (1-based ids)."""
    return os.path.join(directory, f"scene_{scene_id:03d}.{ext}")


def scene_ids(scenes: list[dict], only_scene: int | None) -> list[int]:
    """Scenes to process; validates --scene against the storyboard."""
    ids = [int(s["id"]) for s in scenes]
    if only_scene is None:
        return ids
    if only_scene not in ids:
        bounds = f"{ids[0]}..{ids[-1]}" if ids else "none"
        raise ValueError(
            f"--scene {only_scene} out of range (project has scenes {bounds})")
    return [only_scene]


def _cleanup_tmp(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.unlink(path)
    except OSError:
        pass


def _tmp_output(path: str) -> str:
    """Temp path that keeps the real extension.

    ffmpeg infers the output container from the filename, so
    "scene_001.mp4.tmp" fails while "scene_001.tmp.mp4" works.
    """
    root, ext = os.path.splitext(path)
    return f"{root}.tmp{ext}"


def _interrupt_message(stage: str, scene_id: int, completed: list[int],
                       project: str) -> str:
    done = ", ".join(f"scene_{i:03d}" for i in completed) or "none yet"
    return (f"Generation interrupted during {stage} generation at scene {scene_id}. "
            f"Completed {stage} scenes: {done}. Resume with: "
            f"python ai_video/main.py --project {project} --resume")


def load_project_scenes(base_dir: str) -> list[dict]:
    """Load + structurally validate storyboard.json. Clear errors, no network."""
    storyboard_path = os.path.join(base_dir, "storyboard.json")
    if not os.path.isfile(storyboard_path):
        raise ValueError("Storyboard not found. Generate storyboard first.")
    try:
        with open(storyboard_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Storyboard is corrupt ({storyboard_path}): {exc}")
    try:
        return sb.validate_storyboard(raw)
    except ValueError as exc:
        raise ValueError(f"Storyboard is invalid: {exc}")


def plan_resume_stages(*, scenes: list[dict], img_dir: str, vid_dir: str,
                       aud_dir: str, final_path: str, default_duration: float,
                       stage: str | None = None,
                       only_scene: int | None = None):
    """Decide which stages must run. No writes, no network, no generation.

    Returns (stages_to_run, problems). stages_to_run is a list drawn from
    ["image", "video", "audio", "finalize"]. problems are clear,
    actionable error strings (e.g. missing prerequisite outputs).
    """
    ids = scene_ids(scenes, only_scene)
    by_id = {int(s["id"]): s for s in scenes}

    def images_ok(sid: int) -> bool:
        return st.valid_image_file(scene_path(img_dir, sid, "png"))

    def videos_ok(sid: int) -> bool:
        expected = resolve_scene_duration(by_id[sid], default_duration)
        return st.valid_video_file(scene_path(vid_dir, sid, "mp4"), expected)

    def audios_ok(sid: int) -> bool:
        return st.valid_audio_file(scene_path(aud_dir, sid, "mp3"))

    if stage == "image":
        return ["image"], []
    if stage == "video":
        missing = [f"scene_{sid:03d}" for sid in ids if not images_ok(sid)]
        if missing:
            return [], [f"Cannot run video stage: required image(s) missing or "
                        f"invalid: {', '.join(missing)}. "
                        f"Generate images first (--stage image)."]
        return ["video"], []
    if stage == "audio":
        return ["audio"], []
    if stage == "finalize":
        if only_scene is not None:
            return [], ["--scene does not apply to --stage finalize: "
                        "final.mp4 always covers all scenes in storyboard order."]
        all_ids = sorted(by_id)
        bad_clips = [f"scene_{sid:03d}" for sid in all_ids
                     if not st.valid_video_file(
                         scene_path(vid_dir, sid, "mp4"),
                         resolve_scene_duration(by_id[sid], default_duration))]
        bad_audio = [f"scene_{sid:03d}" for sid in all_ids
                     if not st.valid_audio_file(scene_path(aud_dir, sid, "mp3"))]
        problems = []
        if bad_clips:
            problems.append(
                f"Cannot run finalize stage: video clip(s) missing or "
                f"invalid: {', '.join(bad_clips)}. "
                f"Generate videos first (--stage video).")
        if bad_audio:
            problems.append(
                f"Cannot run finalize stage: narration audio(s) missing or "
                f"invalid: {', '.join(bad_audio)}. "
                f"Generate audio first (--stage audio).")
        if problems:
            return [], problems
        return ["finalize"], []
    # Continue mode: from the first incomplete stage through the end.
    if all(images_ok(sid) for sid in ids):
        if all(videos_ok(sid) for sid in ids):
            if all(audios_ok(sid) for sid in ids):
                if st.valid_final_output(final_path):
                    return [], []
                return ["finalize"], []
            return ["audio", "finalize"], []
        return ["video", "audio", "finalize"], []
    return ["image", "video", "audio", "finalize"], []


def run_image_stage(ctx: dict, scenes: list[dict], only_scene=None,
                    force_all: bool = False) -> dict:
    """Generate keyframes. Skips valid outputs; --scene/force_all regenerate.

    Atomic per scene (tmp + validate + rename). Returns {scene_id: path}.
    """
    provider = ctx["image_provider"]
    state, state_path = ctx["state"], ctx["state_path"]
    by_id = {int(s["id"]): s for s in scenes}
    todo = scene_ids(scenes, only_scene)
    force = force_all or only_scene is not None
    print("[2/5] Generating scene images")
    paths, completed = {}, []
    for sid in todo:
        dest = scene_path(ctx["img_dir"], sid, "png")
        if not force and st.valid_image_file(dest):
            st.mark_scene_complete(state, "image", sid)
            st.save_state(state_path, state)
            print(f"    -> {dest} (reused, valid)")
            paths[sid] = dest
            completed.append(sid)
            continue
        scene = by_id[sid]
        prompt = compose_scene_prompt(
            ctx["subject"], scene["image_prompt"],
            with_reference=ctx["char_ref"] is not None)
        tmp = dest + ".tmp"
        print(f"  scene {sid}/{len(scenes)}: image ...")
        try:
            result = provider.generate(ImageRequest(
                prompt=prompt, negative_prompt=ctx["img_neg"],
                width=ctx["target_w"], height=ctx["target_h"],
                seed=ctx["base_seed"] + sid, output_path=tmp,
                reference_image_path=ctx["char_ref"]))
        except KeyboardInterrupt:
            _cleanup_tmp(tmp)
            raise StageInterrupted(
                _interrupt_message("image", sid, completed, ctx["project"]))
        except Exception:
            _cleanup_tmp(tmp)
            raise
        if not st.valid_image_file(tmp):
            _cleanup_tmp(tmp)
            raise ProviderError(
                f"ImageProvider ({ctx['image_model']}) scene {sid} output "
                f"failed validation")
        os.replace(tmp, dest)
        result.path = Path(dest)
        st.mark_scene_complete(state, "image", sid)
        st.save_state(state_path, state)
        print(f"    -> {dest} ({result.width}x{result.height})")
        paths[sid] = dest
        completed.append(sid)
    return paths


def run_video_stage(ctx: dict, scenes: list[dict], image_paths: dict,
                    only_scene=None, force_all: bool = False) -> dict:
    """Generate clips. Per-scene storyboard durations preserved. Atomic."""
    provider = ctx["video_provider"]
    state, state_path = ctx["state"], ctx["state_path"]
    by_id = {int(s["id"]): s for s in scenes}
    todo = scene_ids(scenes, only_scene)
    force = force_all or only_scene is not None
    print("[3/5] Generating video clips")
    paths, completed = {}, []
    for sid in todo:
        scene = by_id[sid]
        expected = resolve_scene_duration(scene, ctx["default_duration"])
        src = image_paths.get(sid)
        if src is None or not st.valid_image_file(src):
            raise ValueError(
                f"Cannot generate video for scene {sid}: required image "
                f"missing or invalid ({src or 'none'}). Generate images first.")
        dest = scene_path(ctx["vid_dir"], sid, "mp4")
        if not force and st.valid_video_file(dest, expected):
            st.mark_scene_complete(state, "video", sid)
            st.save_state(state_path, state)
            print(f"    -> {dest} (reused, valid)")
            paths[sid] = dest
            completed.append(sid)
            continue
        tmp = dest + ".tmp"
        print(f"  scene {sid}/{len(scenes)}: video ...")
        try:
            result = provider.generate(VideoRequest(
                prompt=scene["video_prompt"], negative_prompt=ctx["vid_neg"],
                input_image=src, width=ctx["target_w"], height=ctx["target_h"],
                duration=expected, seed=ctx["base_seed"] + 1000 + sid,
                output_path=tmp))
        except KeyboardInterrupt:
            _cleanup_tmp(tmp)
            raise StageInterrupted(
                _interrupt_message("video", sid, completed, ctx["project"]))
        except Exception:
            _cleanup_tmp(tmp)
            raise
        if not st.valid_video_file(tmp, expected):
            _cleanup_tmp(tmp)
            raise ProviderError(
                f"VideoProvider ({ctx['video_model']}) scene {sid} output "
                f"failed validation")
        os.replace(tmp, dest)
        result.path = Path(dest)
        st.mark_scene_complete(state, "video", sid)
        st.save_state(state_path, state)
        print(f"    -> {dest} ({result.width}x{result.height}, "
              f"{expected:g}s requested)")
        paths[sid] = dest
        completed.append(sid)
    return paths


def run_audio_stage(ctx: dict, scenes: list[dict], only_scene=None,
                    force_all: bool = False) -> dict:
    """Generate narration mp3s. Skips valid outputs; --scene forces."""
    provider = ctx["audio_provider"]
    state, state_path = ctx["state"], ctx["state_path"]
    by_id = {int(s["id"]): s for s in scenes}
    todo = scene_ids(scenes, only_scene)
    force = force_all or only_scene is not None
    print("[4/5] Generating narration")
    paths, completed = {}, []
    for sid in todo:
        dest = scene_path(ctx["aud_dir"], sid, "mp3")
        if not force and st.valid_audio_file(dest):
            st.mark_scene_complete(state, "audio", sid)
            st.save_state(state_path, state)
            print(f"    -> {dest} (reused, valid)")
            paths[sid] = dest
            completed.append(sid)
            continue
        scene = by_id[sid]
        tmp = dest + ".tmp"
        print(f"  scene {sid}/{len(scenes)}: audio ...")
        try:
            result = provider.generate(AudioRequest(
                text=scene["narration"], output_path=tmp, voice=ctx["voice"],
                rate=ctx["rate"], pitch=ctx["pitch"]))
        except KeyboardInterrupt:
            _cleanup_tmp(tmp)
            raise StageInterrupted(
                _interrupt_message("audio", sid, completed, ctx["project"]))
        except Exception:
            _cleanup_tmp(tmp)
            raise
        if not st.valid_audio_file(tmp):
            _cleanup_tmp(tmp)
            raise ProviderError(
                f"AudioProvider scene {sid} output failed validation")
        os.replace(tmp, dest)
        result.path = Path(dest)
        st.mark_scene_complete(state, "audio", sid)
        st.save_state(state_path, state)
        print(f"    -> {dest}")
        paths[sid] = dest
        completed.append(sid)
    return paths


def run_finalize_stage(ctx: dict, scenes: list[dict], clip_paths: dict,
                       audio_paths: dict) -> str:
    """Mux per-scene clips + concat final.mp4. Verifies every input first."""
    ids = [int(s["id"]) for s in scenes]
    for sid in ids:
        if not st.valid_video_file(clip_paths.get(sid, "")):
            raise ValueError(
                f"Cannot finalize: video clip for scene {sid} missing or "
                f"invalid. Generate videos first.")
        if not st.valid_audio_file(audio_paths.get(sid, "")):
            raise ValueError(
                f"Cannot finalize: narration for scene {sid} missing or "
                f"invalid. Generate audio first.")
    print("[5/5] Rendering final video")
    muxed = []
    try:
        for sid in ids:
            out = scene_path(ctx["mux_dir"], sid, "mp4")
            tmp = _tmp_output(out)
            ff.mux_scene(clip_paths[sid], audio_paths[sid], tmp,
                         executable=ctx["ffmpeg_exe"], copy_video=True)
            if not st.valid_final_output(tmp):
                _cleanup_tmp(tmp)
                raise ProviderError(f"Finalize: muxed scene {sid} invalid "
                                    f"(missing video/audio stream)")
            os.replace(tmp, out)
            muxed.append(out)
        final_tmp = _tmp_output(ctx["final_path"])
        ff.concat_scenes(muxed, final_tmp, executable=ctx["ffmpeg_exe"], copy=True)
        if not st.valid_final_output(final_tmp):
            _cleanup_tmp(final_tmp)
            raise ProviderError("Finalize: final.mp4 failed validation "
                                "(missing video/audio stream)")
        os.replace(final_tmp, ctx["final_path"])
    except KeyboardInterrupt:
        for i in ids:
            _cleanup_tmp(_tmp_output(scene_path(ctx["mux_dir"], i, "mp4")))
        _cleanup_tmp(_tmp_output(ctx["final_path"]))
        done = [i for i in ids if st.valid_video_file(
            scene_path(ctx["mux_dir"], i, "mp4"))]
        raise StageInterrupted(
            f"Generation interrupted during finalization. Completed muxed "
            f"scenes: {done or 'none'}. Resume with: "
            f"python ai_video/main.py --project {ctx['project']} --resume")
    st.mark_stage_complete(ctx["state"], "finalize")
    st.save_state(ctx["state_path"], ctx["state"])
    print(f"Done -> {ctx['final_path']}")
    return ctx["final_path"]


def _resolve_char_ref(character_reference: str | None) -> str | None:
    """Resolve --character-reference to an existing file. Clear error if not."""
    if character_reference is None:
        return None
    path = os.path.abspath(character_reference)
    if not os.path.isfile(path):
        raise ValueError(f"--character-reference not found: {path}")
    return path


def run_fresh_flow(args, ctx, cfg, ollama_url, ollama_model, num_scenes,
                   stages_planned):
    """Full generation from scratch. Always rebuilds state (explicit rerun)."""
    ollama_cfg = cfg.get("ollama", {})
    state, state_path = ctx["state"], ctx["state_path"]
    fresh = st.new_state(ctx["project"])
    fresh["base_seed"] = ctx["base_seed"]
    ctx["state"] = fresh
    state = fresh
    print("[1/5] Generating storyboard")
    research_context, research_meta = maybe_research(args.research, args.prompt)
    if research_meta is not None:
        quality = research_meta.get("research_quality", "?")
        print(f"  Research: web ({len(research_meta['sources'])} sources, "
              f"quality={quality})")
    raw = ol.generate_storyboard(args.prompt, model=ollama_model,
                                 base_url=ollama_url, num_scenes=num_scenes,
                                 timeout=int(ollama_cfg.get("timeout", 180)),
                                 research_context=research_context)
    scenes = sb.validate_storyboard(raw, research_meta)
    rewrite_timeout = int(ollama_cfg.get("timeout", 180))

    def rewrite_scene(scene, cited_facts, reason):
        return ol.rewrite_scene_narration(
            scene, cited_facts, reason, model=ollama_model,
            base_url=ollama_url, timeout=rewrite_timeout)

    scenes, rewrite_count, sem_warnings = ensure_grounded_scenes(
        scenes, research_meta, rewrite_scene)
    for warning in sem_warnings:
        print(f"  Semantic warning: {warning}")
    if rewrite_count:
        print(f"  Rewrote {rewrite_count} scene narration(s) for grounding.")
    subject = SubjectProfile.from_dict(
        raw.get("subject") if isinstance(raw, dict) else None)
    sb.save_storyboard(scenes, ctx["storyboard_path"], subject, research_meta)
    state["scene_ids"] = [int(s["id"]) for s in scenes]
    st.mark_stage_complete(state, "research")
    st.mark_stage_complete(state, "storyboard")
    st.save_state(state_path, state)
    print(f"  Saved {len(scenes)} scenes -> {ctx['storyboard_path']}")
    if stages_planned == ["storyboard"]:
        print("Stopped after storyboard as requested (--stop-after storyboard).")
        return 0
    if subject is not None:
        print(f"  Subject profile: {subject.subject_type or 'unspecified type'}")

    ctx["char_ref"] = _resolve_char_ref(args.character_reference)
    ctx["subject"] = subject
    image_paths, clip_paths, audio_paths = {}, {}, {}
    if "image" in stages_planned:
        image_paths = run_image_stage(ctx, scenes, force_all=True)
        if "video" not in stages_planned and "audio" not in stages_planned \
                and "finalize" not in stages_planned:
            print(f"Stopped after images as requested. "
                  f"{len(image_paths)} scene images in {ctx['img_dir']}.")
            return 0
    if "video" in stages_planned:
        clip_paths = run_video_stage(ctx, scenes, image_paths, force_all=True)
    if "audio" in stages_planned:
        audio_paths = run_audio_stage(ctx, scenes, force_all=True)
        if "finalize" not in stages_planned:
            print("Stopped after audio as requested (--stop-after audio). "
                  f"{len(audio_paths)} narration files in {ctx['aud_dir']}.")
            return 0
    if "finalize" in stages_planned:
        run_finalize_stage(ctx, scenes, clip_paths, audio_paths)
    return 0


def run_resume_flow(args, ctx, scenes, stages_planned):
    """Continue an existing project. Never touches research/storyboard/Ollama."""
    storyboard_path = ctx["storyboard_path"]
    subject = sb.load_subject(storyboard_path)
    ctx["subject"] = subject
    if subject is not None:
        print(f"  Subject profile: {subject.subject_type or 'unspecified type'}")
    if "image" in stages_planned:
        ctx["char_ref"] = _resolve_char_ref(args.character_reference)
    only = args.scene
    ids = [int(s["id"]) for s in scenes]
    if "image" in stages_planned:
        run_image_stage(ctx, scenes, only, force_all=False)
    if "video" in stages_planned:
        full_images = {sid: scene_path(ctx["img_dir"], sid, "png") for sid in ids}
        run_video_stage(ctx, scenes, full_images, only, force_all=False)
    if "audio" in stages_planned:
        run_audio_stage(ctx, scenes, only, force_all=False)
    if "finalize" in stages_planned:
        full_clips = {sid: scene_path(ctx["vid_dir"], sid, "mp4") for sid in ids}
        full_audios = {sid: scene_path(ctx["aud_dir"], sid, "mp3") for sid in ids}
        run_finalize_stage(ctx, scenes, full_clips, full_audios)
    if args.stage is not None:
        if args.stage == "finalize":
            scope = "all scenes"
        elif only is not None:
            scope = f"scene {only}"
        else:
            scope = "missing scenes"
        print(f"Stage '{args.stage}' complete ({scope}).")
    return 0


def maybe_research(research_mode: str, topic: str) -> tuple[str | None, dict | None]:
    """Run web research when enabled. Returns (context, metadata).

    Mode "none" returns (None, None) without touching the research
    provider (no provider instantiation, no network). Mode "web" collects
    sources and renders labeled Ollama context plus persistable metadata.
    Failures raise (never fake info); main() reports them.
    """
    if research_mode == "none":
        return None, None
    provider = create_provider("research", APP_DIR, research_mode)
    result = provider.research(topic)
    metadata = {"mode": "web",
                "sources": [s.to_metadata() for s in result.sources]}
    pack = getattr(result, "pack", None)
    if pack is not None:
        metadata.update({"query": pack.query,
                         "research_quality": pack.research_quality,
                         "retrieved_at": pack.retrieved_at,
                         "facts": [f.to_record() for f in pack.facts],
                         "conflicts": [c.to_record() for c in pack.conflicts],
                         "source_records": [s.to_record() for s in pack.sources],
                         "diagnostics": list(getattr(pack, "diagnostics", []))})
    return result.render_context(), metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal AI video generator (provider architecture).")
    p.add_argument("prompt", nargs="?", default=None,
                   help="Video idea (required unless --resume is used)")
    p.add_argument("--project", default=None, help="Project name (default: derived from prompt)")
    p.add_argument("--image-model", default="sd15", help="Image model from config/models.json")
    p.add_argument("--video-model", default="ltx", help="Video model from config/models.json")
    p.add_argument("--audio-model", default="edge_tts", help="Audio provider/model, e.g. edge or piper (see config/models.json)")
    p.add_argument("--aspect", default="16:9", help="Target aspect, e.g. 16:9 or 9:16")
    p.add_argument("--resolution", default=720, help="Target resolution, e.g. 420 or 720")
    p.add_argument("--character-reference", default=None,
                   help="Optional character reference image reused for every scene "
                        "(passed through to the image provider; provider decides support)")
    p.add_argument("--stop-after", default="all",
                   choices=["storyboard", "image", "images", "video", "videos",
                            "audio", "all"],
                   help="Stop the pipeline after the given stage (default: all)")
    p.add_argument("--stage", default=None, choices=["image", "video", "audio",
                                                   "finalize"],
                   help="Run only one stage from an existing project (requires --resume)")
    p.add_argument("--scene", default=None, type=int,
                   help="Limit the requested stage(s) to one 1-based scene number "
                        "(requires --resume; regenerates that scene even if valid)")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing project: load storyboard.json, skip "
                        "complete stages, never regenerate the storyboard")
    p.add_argument("--research", default="none", choices=["none", "web"],
                   help="Research mode: none (default, fully offline) or web "
                        "(research the topic before storyboarding)")
    # Infra overrides (backward compatible with the pre-provider CLI).
    p.add_argument("--model", default=None, help="Ollama model (default from config.yaml)")
    p.add_argument("--voice", default=None, help="Voice for the audio provider: Edge voice name, or Piper voice file stem (default from config.yaml)")
    p.add_argument("--config", default=os.path.join(APP_DIR, "config.yaml"))
    p.add_argument("--scenes", type=int, default=None, help="Number of scenes")
    p.add_argument("--comfy-url", default=None, help="ComfyUI URL override")
    p.add_argument("--ollama-url", default=None, help="Ollama URL override")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    combo_error = validate_cli_combination(args)
    if combo_error is not None:
        print(f"Error: {combo_error}", file=sys.stderr)
        return 1
    cfg = load_config(args.config)

    ollama_cfg = cfg.get("ollama", {})
    comfy_cfg = cfg.get("comfyui", {})
    tts_cfg = cfg.get("tts", {})
    ff_cfg = cfg.get("ffmpeg", {})
    pipe_cfg = cfg.get("pipeline", {})

    ollama_url = args.ollama_url or ollama_cfg.get("url", "http://127.0.0.1:11434")
    ollama_model = args.model or ollama_cfg.get("model", "qwen3:8b")
    num_scenes = args.scenes or int(ollama_cfg.get("num_scenes", 3))
    comfy_url = args.comfy_url or comfy_cfg.get("url", "http://127.0.0.1:8188")
    voice = args.voice or tts_cfg.get("voice", "en-US-AriaNeural")
    default_duration = float(pipe_cfg.get("duration", 5))
    stop_after = normalize_stop_after(args.stop_after)
    if args.resume:
        if args.prompt is not None:
            print("Note: prompt is ignored in resume mode; "
                  "using the existing storyboard.", file=sys.stderr)
        project = slugify(args.project)
    else:
        project = slugify(args.project) if args.project else slugify(args.prompt[:40])
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    try:
        target_w, target_h = resolve_dimensions(args.aspect, args.resolution)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Validate requested models against the registry before doing any work.
    try:
        img_entry = lookup(APP_DIR, "image", args.image_model)
        vid_entry = lookup(APP_DIR, "video", args.video_model)
        aud_entry = lookup(APP_DIR, "audio", args.audio_model)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Image model: {args.image_model}")
    print(f"Video model: {args.video_model}")
    print(f"Aspect: {args.aspect}")
    print(f"Resolution: {args.resolution}p")
    print(f"ComfyUI: {comfy_url}")
    print(f"Research: {args.research}")
    print(f"Target: {target_w}x{target_h}")

    base_dir = os.path.join(APP_DIR, "projects", project)
    img_dir = os.path.join(base_dir, "images")
    vid_dir = os.path.join(base_dir, "videos")
    aud_dir = os.path.join(base_dir, "audio")
    mux_dir = os.path.join(base_dir, "muxed")
    for d in (img_dir, vid_dir, aud_dir, mux_dir):
        os.makedirs(d, exist_ok=True)
    storyboard_path = os.path.join(base_dir, "storyboard.json")
    state_path = os.path.join(base_dir, "state.json")
    final_path = os.path.join(base_dir, "final.mp4")

    resume_mode = bool(args.resume)
    if resume_mode:
        if not os.path.isdir(base_dir):
            print(f"Error: --resume: project not found: {project}. "
                  f"Generate it first (without --resume).", file=sys.stderr)
            return 1
        try:
            resume_scenes = load_project_scenes(base_dir)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.scene is not None:
            try:
                scene_ids(resume_scenes, args.scene)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        stages_planned, resume_problems = plan_resume_stages(
            scenes=resume_scenes, img_dir=img_dir, vid_dir=vid_dir,
            aud_dir=aud_dir, final_path=final_path,
            default_duration=default_duration, stage=args.stage,
            only_scene=args.scene)
        if resume_problems:
            print("Error: " + "\nError: ".join(resume_problems), file=sys.stderr)
            return 1
        # --stop-after still gates which of the planned stages run.
        order = ["image", "video", "audio", "finalize"]
        if stop_after != "all":
            stages_planned = [s for s in stages_planned
                              if order.index(s) <= order.index(stop_after)]
        if not stages_planned:
            print(f"Project {project} is already complete through "
                  f"'{stop_after}'. Nothing to do.")
            return 0
    else:
        stages_planned = ["storyboard", "image", "video", "audio", "finalize"]
        if stop_after != "all":
            full_order = ["storyboard", "image", "video", "audio", "finalize"]
            stages_planned = [s for s in full_order
                              if full_order.index(s) <= full_order.index(stop_after)]

    try:
        if not resume_mode:
            ol.check_reachable(ollama_url)
        client = ComfyClient(
            comfy_url,
            timeout=int(comfy_cfg.get("timeout", 60)),
            poll_interval=float(comfy_cfg.get("poll_interval", 2.0)),
            queue_timeout=int(comfy_cfg.get("queue_timeout", 5400)))
        if "image" in stages_planned or "video" in stages_planned:
            client.check_reachable()
        ffmpeg_exe = None
        if "finalize" in stages_planned:
            ffmpeg_exe = ff.check_ffmpeg(ff_cfg.get("executable", "ffmpeg"))
    except (RuntimeError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        image_provider = create_provider("image", APP_DIR, args.image_model, client=client)
        video_provider = create_provider("video", APP_DIR, args.video_model, client=client)
        audio_provider = create_provider("audio", APP_DIR, args.audio_model)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    img_neg = img_entry.get("negative_prompt", "")
    vid_neg = vid_entry.get("negative_prompt", "")

    state = st.load_state(state_path, project)
    if args.seed is not None:
        base_seed = args.seed
    elif state.get("base_seed") is not None and resume_mode:
        base_seed = int(state["base_seed"])

    ctx = {
        "project": project, "base_dir": base_dir,
        "storyboard_path": storyboard_path,
        "img_dir": img_dir, "vid_dir": vid_dir,
        "aud_dir": aud_dir, "mux_dir": mux_dir, "final_path": final_path,
        "state_path": state_path, "state": state,
        "image_provider": image_provider, "video_provider": video_provider,
        "audio_provider": audio_provider,
        "image_model": args.image_model, "video_model": args.video_model,
        "target_w": target_w, "target_h": target_h, "base_seed": base_seed,
        "img_neg": img_neg, "vid_neg": vid_neg,
        "voice": voice, "rate": str(tts_cfg.get("rate", "+0%")),
        "pitch": str(tts_cfg.get("pitch", "+0Hz")),
        "default_duration": default_duration, "char_ref": None,
        "subject": None, "ffmpeg_exe": ffmpeg_exe,
    }

    try:
        if resume_mode:
            return run_resume_flow(args, ctx, resume_scenes, stages_planned)
        return run_fresh_flow(args, ctx, cfg, ollama_url, ollama_model,
                              num_scenes, stages_planned)
    except StageInterrupted as exc:
        print(exc.message, file=sys.stderr)
        return 130
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exception(exc.__cause__ or exc)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
