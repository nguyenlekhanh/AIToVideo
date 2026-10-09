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
import promptfiles as pf
import storyboard as sb
import state as st
import continuous as ch
import script as sc
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
    if args.prompt is None and not args.resume and not getattr(args, "keyframes", None) and not getattr(args, "script", None) and not getattr(args, "prompt_dir", None):
        return "the prompt is required unless --resume, --keyframes, --script or --prompt-dir is used"
    continuous = bool(getattr(args, "continuous", False))
    if not continuous:
        if getattr(args, "image", None) is not None:
            return "--image requires --continuous"
        if getattr(args, "clips", None) is not None:
            return "--clips requires --continuous"
        if getattr(args, "duration", None) is not None:
            return "--duration requires --continuous"
        if getattr(args, "identity_file", None) is not None:
            return "--identity-file requires --continuous"
    if continuous:
        if getattr(args, "keyframes", None) or getattr(args, "script", None):
            return "cannot use --keyframes/--script with --continuous (choose one input mode)"
        if args.resume:
            return "cannot use --resume with --continuous (re-running the same command resumes the chain)"
        if args.stage is not None:
            return "cannot use --stage with --continuous (use --scene N to regenerate one clip)"
        if getattr(args, "character_reference", None):
            return "--character-reference is not used in --continuous mode"
        if not getattr(args, "image", None):
            return "--continuous requires --image (reference/start image)"
        if getattr(args, "scenes", None) is not None:
            return "use --clips (not --scenes) with --continuous"
        clips = getattr(args, "clips", None)
        duration = getattr(args, "duration", None)
        if clips is not None and clips < 1:
            return "--clips must be a positive clip count"
        if duration is not None and duration <= 0:
            return "--duration must be a positive number of seconds"
        if clips is not None and duration is not None:
            if not 3 * clips <= duration <= 15 * clips:
                return (f"--duration {duration:g}s is infeasible for "
                        f"--clips {clips} (each clip is 3..15s; need "
                        f"{3 * clips}..{15 * clips}s)")
        if normalize_stop_after(args.stop_after) not in ("all", "storyboard"):
            return "--continuous only supports --stop-after storyboard|all"
        return None
    if getattr(args, "prompt", None) and getattr(args, "keyframes", None):
        return "cannot use --keyframes with a prompt"
    if getattr(args, "prompt", None) and getattr(args, "script", None):
        return "cannot use --script with a prompt"
    if getattr(args, "keyframes", None) and getattr(args, "script", None):
        return "cannot use --script with --keyframes (choose one input mode)"
    if getattr(args, "keyframes", None) and args.resume:
        return "cannot use --keyframes with --resume (resume loads the existing storyboard)"
    if getattr(args, "script", None) and args.resume:
        return "cannot use --script with --resume (resume loads the existing storyboard)"
    if getattr(args, "keyframes", None) and not getattr(args, "project", None):
        return "--keyframes requires --project"
    if getattr(args, "script", None) and not getattr(args, "project", None):
        return "--script requires --project"
    if args.resume and not args.project:
        return "--resume requires --project"
    if args.stage is not None and not args.resume:
        return "--stage requires --resume"
    if args.scene is not None and not args.resume and not bool(getattr(args, "continuous", False)):
        return "--scene requires --resume"
    if args.scene is not None and args.scene < 1:
        return "--scene must be a positive 1-based scene number"
    if args.stage is not None and normalize_stop_after(args.stop_after) != "all":
        return "--stage and --stop-after are mutually exclusive"
    if getattr(args, "video_mode", "i2v") == "t2v" \
            and getattr(args, "video_model", "ltx") != "ltx":
        return "--video-mode t2v is only supported by --video-model ltx"
    prompt_dir = getattr(args, "prompt_dir", None)
    if prompt_dir:
        if not getattr(args, "project", None):
            return "--prompt-dir requires --project"
        if args.prompt is not None:
            return "cannot use a prompt with --prompt-dir (prompt files are the source)"
        if getattr(args, "keyframes", None) or getattr(args, "script", None):
            return "cannot use --keyframes/--script with --prompt-dir (choose one input mode)"
        if bool(getattr(args, "continuous", False)):
            return "cannot use --continuous with --prompt-dir (choose one input mode)"
        if getattr(args, "scenes", None) is not None:
            return "cannot use --scenes with --prompt-dir (clip count comes from prompt files)"
        if args.stage == "image":
            return "cannot use --stage image with --prompt-dir (prompt files provide video prompts only)"
        if normalize_stop_after(args.stop_after) in ("storyboard", "image"):
            return "--stop-after storyboard|image(s) cannot be used with --prompt-dir"
    if bool(getattr(args, "use_first_frame", False)):
        if not prompt_dir:
            return "--use-first-frame requires --prompt-dir"
        if getattr(args, "video_model", "ltx") != "minimax_h3":
            return "--use-first-frame is only supported by --video-model minimax_h3"
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


def ensure_clean_visuals(scenes, rewrite_fn, max_rounds=2):
    """Bounded regeneration of image/video prompts that request rendered
    text/numbers or multi-panel compositions.

    scenes: validated scene list (mutated in place on rewrite). rewrite_fn
    takes (scene, reason) and returns an updated scene with new prompts
    (narration, ids, grounding and fact/source ids preserved). Returns
    (scenes, rewrite_count). Raises ValueError when prompts are still
    unclean after max_rounds. Research grounding is never touched here.
    """
    rewrites = 0
    for _ in range(max_rounds + 1):
        failing: dict[int, str] = {}
        for i, scene in enumerate(scenes, start=1):
            problems = sb.validate_visual_prompts([scene])
            if problems:
                failing[i] = problems[0]
        if not failing:
            return scenes, rewrites
        if rewrites >= max_rounds:
            raise ValueError(
                "Storyboard visual prompts still request rendered "
                "text/statistics or multi-panel layouts after "
                f"{max_rounds} rewrites:\n- "
                + "\n- ".join(failing.values()))
        for index in sorted(failing):
            scenes[index - 1] = rewrite_fn(
                dict(scenes[index - 1]), failing[index])
            rewrites += 1
    raise AssertionError("unreachable")  # loop always returns or raises above


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


def prepare_prompt_scenes(prompt_dir_arg: str, base_dir: str,
                          default_duration: float, project: str) -> list[dict]:
    """Discover prompt files and build one scene per file (verbatim).

    Relative directories resolve against the project directory, so
    --prompt-dir prompt with --project news1 reads
    projects/news1/prompt/. Raises FileNotFoundError/ValueError with
    clear messages (reported by main, never silent).
    """
    prompt_dir = (prompt_dir_arg if os.path.isabs(prompt_dir_arg)
                  else os.path.join(base_dir, prompt_dir_arg))
    print(f"[Video Prompts] Project: {project}")
    print(f"[Video Prompts] Directory: {prompt_dir}")
    entries, ignored = pf.discover_prompt_files(prompt_dir)
    for name in ignored:
        print(f"[Video Prompts] Ignoring non-clip file: {name}")
    print(f"[Video Prompts] Found {len(entries)} prompt file(s): "
          + ", ".join(os.path.basename(p) for _, p in entries))
    manifest = pf.load_manifest(prompt_dir)
    if manifest is not None:
        print(f"[Video Prompts] Manifest: {pf.MANIFEST_FILENAME} "
              f"(per-clip durations: "
              + ", ".join(f"{n}->{manifest[n]:g}s"
                          for n in sorted(manifest)) + ")")
    else:
        print(f"[Video Prompts] No manifest (Version 1 durations: "
              f"{default_duration:g}s per clip)")
    scenes = pf.load_prompt_scenes(prompt_dir, default_duration)
    return scenes


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
                       only_scene: int | None = None,
                       video_needs_images: bool = True,
                       check_video_duration: bool = True):
    """Decide which stages must run. No writes, no network, no generation.

    Returns (stages_to_run, problems). stages_to_run is a list drawn from
    ["image", "video", "audio", "finalize"]. problems are clear,
    actionable error strings (e.g. missing prerequisite outputs).

    video_needs_images=False skips the scene-image prerequisite (motion
    backends with a pinned reference image). check_video_duration=False
    validates clips by existence/readability only (fixed-length backends
    such as wan_animate2).
    """
    ids = scene_ids(scenes, only_scene)
    by_id = {int(s["id"]): s for s in scenes}

    def images_ok(sid: int) -> bool:
        return st.valid_image_file(scene_path(img_dir, sid, "png"))

    def videos_ok(sid: int) -> bool:
        expected = (resolve_scene_duration(by_id[sid], default_duration)
                    if check_video_duration else None)
        return st.valid_video_file(scene_path(vid_dir, sid, "mp4"), expected)

    def audios_ok(sid: int) -> bool:
        return st.valid_audio_file(scene_path(aud_dir, sid, "mp3"))

    if stage == "image":
        return ["image"], []
    if stage == "video":
        if video_needs_images:
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
        def clip_ok(sid: int) -> bool:
            expected = (resolve_scene_duration(by_id[sid], default_duration)
                        if check_video_duration else None)
            return st.valid_video_file(scene_path(vid_dir, sid, "mp4"), expected)
        bad_clips = [f"scene_{sid:03d}" for sid in all_ids if not clip_ok(sid)]
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
    # Backends that need no input images (t2v, pinned motion reference)
    # skip the image prerequisite entirely.
    if (not video_needs_images) or all(images_ok(sid) for sid in ids):
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


def chain_first_frame_upload(ctx: dict, scenes: list[dict], sid: int) -> str | None:
    """Resolve the first-frame upload name for clip `sid` (or None).

    First clip in scene order stays pure T2V. Any later clip requires the
    previous scene's clip to exist and validate; its last frame is
    extracted (existing ffmpeg utility, project-local continuity dir) and
    uploaded with the existing ComfyUI client. Clear errors, no silent
    T2V fallback. The workflow's last-frame input is never involved.
    """
    provider = ctx["video_provider"]
    by_id = {int(s["id"]): s for s in scenes}
    ordered = sorted(by_id)
    if sid == ordered[0]:
        print(f"  First frame: none (first clip)")
        return None
    prev = max(i for i in ordered if i < sid)
    prev_scene = by_id[prev]
    prev_expected = resolve_scene_duration(prev_scene, ctx["default_duration"])
    prev_path = scene_path(ctx["vid_dir"], prev, "mp4")
    if not st.valid_video_file(prev_path, prev_expected):
        raise ValueError(
            f"Cannot generate video for scene {sid}: first-frame source "
            f"scene_{prev:03d}.mp4 missing or invalid. Generate it first.")
    continuity_dir = os.path.join(ctx["base_dir"], "continuity")
    frame_path = os.path.join(continuity_dir, f"scene_{prev:03d}_last.png")
    print(f"  First frame source: {prev_path}")
    print(f"  Extracting last frame...")
    ff.extract_last_frame(prev_path, frame_path,
                          executable=ctx.get("ffmpeg_exe") or "ffmpeg")
    print(f"  Uploading first frame to ComfyUI...")
    return provider.client.upload_image(frame_path)


def run_video_stage(ctx: dict, scenes: list[dict], image_paths: dict,
                    only_scene=None, force_all: bool = False) -> dict:
    """Generate clips. Per-scene storyboard durations preserved. Atomic.

    Motion backends (wan_animate2 + --motion-dir): one driving clip per
    scene (001 -> scene 1, ...), same reference image for every job when
    --reference-image is pinned, fixed-length outputs validated by
    existence/readability instead of storyboard duration.
    """
    provider = ctx["video_provider"]
    state, state_path = ctx["state"], ctx["state_path"]
    by_id = {int(s["id"]): s for s in scenes}
    todo = scene_ids(scenes, only_scene)
    force = force_all or only_scene is not None
    motion_clips = (ctx.get("motion_clips")
                    if getattr(provider, "uses_motion_clips", False) else None)
    pinned_ref = (ctx.get("reference_image")
                  if getattr(provider, "supports_pinned_reference", False)
                  else None)
    skip_duration = bool(getattr(provider, "fixed_duration", False))
    needs_image = bool(getattr(provider, "needs_input_image", True))
    chain = bool(ctx.get("first_frame_chain", False)) and bool(
        getattr(provider, "supports_first_frame", False))
    if ctx.get("first_frame_chain", False) and not getattr(
            provider, "supports_first_frame", False):
        raise ValueError(
            f"--use-first-frame is not supported by video model "
            f"'{ctx.get('video_model')}'.")
    if chain:
        print("First-frame chaining: enabled")
    if motion_clips is not None:
        print(f"  Motion clips: {len(motion_clips)} "
              f"({os.path.basename(os.path.dirname(motion_clips[0]))}/)")
        if len(motion_clips) > len(scenes):
            print(f"  Note: ignoring {len(motion_clips) - len(scenes)} extra "
                  f"motion clip(s) beyond {len(scenes)} scenes.")
    if pinned_ref is not None:
        print(f"  Reference image (pinned for all jobs): {pinned_ref}")
    print("[3/5] Generating video clips")
    paths, completed = {}, []
    for sid in todo:
        scene = by_id[sid]
        expected = resolve_scene_duration(scene, ctx["default_duration"])
        check_duration = None if skip_duration else expected
        if pinned_ref is not None:
            src = pinned_ref
        elif not needs_image:
            src = None
        else:
            src = image_paths.get(sid)
            if src is None or not st.valid_image_file(src):
                raise ValueError(
                    f"Cannot generate video for scene {sid}: required image "
                    f"missing or invalid ({src or 'none'}). Generate images first.")
        motion = None
        if motion_clips is not None:
            if sid - 1 >= len(motion_clips):
                raise ValueError(
                    f"Cannot generate video for scene {sid}: only "
                    f"{len(motion_clips)} motion clip(s) in "
                    f"{ctx.get('motion_dir', 'motion-dir')}; scenes are "
                    f"mapped 001 -> scene 1, 002 -> scene 2, ...")
            motion = motion_clips[sid - 1]
        dest = scene_path(ctx["vid_dir"], sid, "mp4")
        if not force and st.valid_video_file(dest, check_duration):
            st.mark_scene_complete(state, "video", sid)
            st.save_state(state_path, state)
            print(f"    -> {dest} (reused, valid)")
            paths[sid] = dest
            completed.append(sid)
            continue
        tmp = dest + ".tmp"
        first_frame = chain_first_frame_upload(ctx, scenes, sid) if chain else None
        if motion is not None:
            print(f"  scene {sid}/{len(scenes)}: video "
                  f"(ref={os.path.basename(str(src))}, "
                  f"motion={os.path.basename(motion)}) ...")
        elif not needs_image:
            extra = f", prompt={scene['prompt_file']}" if scene.get("prompt_file") else ""
            print(f"  scene {sid}/{len(scenes)}: video (t2v, no input image{extra}) ...")
        else:
            print(f"  scene {sid}/{len(scenes)}: video ...")
        try:
            result = provider.generate(VideoRequest(
                prompt=scene["video_prompt"], negative_prompt=ctx["vid_neg"],
                input_image=src, width=ctx["target_w"], height=ctx["target_h"],
                duration=expected, seed=ctx["base_seed"] + 1000 + sid,
                output_path=tmp, motion_video=(Path(motion) if motion else None),
                first_frame=first_frame))
        except KeyboardInterrupt:
            _cleanup_tmp(tmp)
            raise StageInterrupted(
                _interrupt_message("video", sid, completed, ctx["project"]))
        except Exception:
            _cleanup_tmp(tmp)
            raise
        if not st.valid_video_file(tmp, check_duration):
            _cleanup_tmp(tmp)
            raise ProviderError(
                ctx["video_model"],
                f"VideoProvider ({ctx['video_model']}) scene {sid} output "
                f"failed validation")
        os.replace(tmp, dest)
        result.path = Path(dest)
        st.mark_scene_complete(state, "video", sid)
        st.save_state(state_path, state)
        if skip_duration:
            print(f"    -> {dest} ({result.width}x{result.height}, "
                  f"fixed-length backend output)")
        else:
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
    if getattr(args, "keyframes", None):
        # Mode B: keyframes -> Qwen-VL -> storyboard. No text prompt, no
        # research topic; the images are the source. Same save/validate/
        # resume path as the text flow below.
        analysis_model = getattr(args, "analysis_model", None) or "qwen3-vl:8b"
        raw = ol.generate_storyboard_from_keyframes(
            args.keyframes, model=analysis_model, base_url=ollama_url,
            num_scenes=args.scenes,
            timeout=int(ollama_cfg.get("timeout", 600)))
        research_meta = None
    elif getattr(args, "script", None):
        # Mode C: long source text -> Qwen condensation -> video script
        # -> storyboard. Text-only Qwen calls; master subject profile
        # persisted with the storyboard for all downstream stages.
        analysis_model = getattr(args, "analysis_model", None) or "qwen3:8b"
        base_dir = os.path.join(APP_DIR, "projects", ctx["project"])
        script_path = os.path.join(base_dir, args.script);
        raw, condensed, source_text = sc.generate_from_script(
            script_path, model=analysis_model, base_url=ollama_url,
            num_scenes=args.scenes,
            timeout=int(ollama_cfg.get("timeout", 600)),
            subject=getattr(args, "subject", "auto"))
        with open(os.path.join(ctx["base_dir"], "source.txt"), "w",
                  encoding="utf-8") as f:
            f.write(source_text)
        sc.save_condensed_script(
            condensed, os.path.join(ctx["base_dir"], "condensed_script.json"))
        print(f"  Saved condensed script -> "
              f"{os.path.join(ctx['base_dir'], 'condensed_script.json')}")
        research_meta = None
    else:
        research_context, research_meta = maybe_research(args.research, args.prompt)
        if research_meta is not None:
            quality = research_meta.get("research_quality", "?")
            print(f"  Research: web ({len(research_meta['sources'])} sources, "
                  f"quality={quality})")
        print("[Storyboard] planner starting "
              f"(research quality={research_meta.get('research_quality', 'none') if research_meta else 'none'})")
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

    def rewrite_visuals(scene, reason):
        return ol.rewrite_scene_visuals(
            scene, reason, model=ollama_model,
            base_url=ollama_url, timeout=rewrite_timeout)

    scenes, visual_rewrites = ensure_clean_visuals(scenes, rewrite_visuals)
    if visual_rewrites:
        print(f"  Rewrote {visual_rewrites} scene visual prompt(s) for "
              f"clean single-frame generation.")
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


def run_prompt_fresh_flow(args, ctx, stages_planned):
    """Fresh prompt-file run: scenes from prompt/*.txt, no storyboard,
    research, planner, or Ollama. Shares the video/audio/finalize stages
    (resume, --scene, validation, finalize discovery all unchanged)."""
    state, state_path = ctx["state"], ctx["state_path"]
    fresh = st.new_state(ctx["project"])
    fresh["base_seed"] = ctx["base_seed"]
    ctx["state"] = fresh
    state = fresh
    scenes = prepare_prompt_scenes(args.prompt_dir, ctx["base_dir"],
                                   ctx["default_duration"], ctx["project"])
    state["scene_ids"] = [int(s["id"]) for s in scenes]
    st.save_state(state_path, state)
    # Prompt files are video-only: storyboard/image stages never apply.
    stages_planned = [s for s in stages_planned
                      if s not in ("storyboard", "image")]
    ctx["char_ref"] = None
    ctx["subject"] = None
    image_paths, clip_paths, audio_paths = {}, {}, {}
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
    if ctx.get("prompt_mode"):
        # Prompt-file mode: scenes already come from prompt/*.txt verbatim;
        # there is no storyboard.json to load a subject from.
        subject = None
    else:
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
                   help="Video idea (required unless --resume, --keyframes or --script is used)")
    p.add_argument("--project", default=None, help="Project name (default: derived from prompt)")
    p.add_argument("--image-model", default="sd15", help="Image model from config/models.json")
    p.add_argument("--video-model", default="ltx", help="Video model from config/models.json")
    p.add_argument("--video-mode", default="i2v", choices=["i2v", "t2v"],
                   help="Video generation mode: i2v = image-to-video (default), "
                        "t2v = text-to-video, LTX only, no input image")
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
    p.add_argument("--keyframes", default=None,
                   help="Directory containing source keyframe images for visual analysis")
    p.add_argument("--analysis-model", dest="analysis_model",
                   default=argparse.SUPPRESS,
                   help="Ollama model used for storyboard analysis "
                        "(defaults: qwen3-vl:8b with --keyframes, "
                        "qwen3:8b with --script or --continuous)")
    p.add_argument("--analysis", dest="analysis_model",
                   default=argparse.SUPPRESS,
                   help="Alias for --analysis-model")
    p.add_argument("--script", default=None,
                   help="Source text file for Mode C script-to-storyboard "
                        "(long text is condensed by Qwen first)")
    p.add_argument("--subject", default="auto",
                   help="Master subject profile for Mode C: 'auto' (derived "
                        "from the source) or a preset (none, god)")
    p.add_argument("--reference-image", default=None,
                   help="Single reference image reused for every wan_animate2 "
                        "motion job (defaults to each scene's own image)")
    p.add_argument("--motion-dir", default=None,
                   help="Directory of driving motion clips for "
                        "--video-model wan_animate2 (001.mp4, 002.mp4, ...)")
    p.add_argument("--prompt-dir", default=None,
                   help="Project-local directory of numbered video prompt "
                        "files (1.txt, 2.txt, ...); relative paths resolve "
                        "against projects/<project>/. Each file is one clip's "
                        "complete video prompt (no storyboard/research).")
    p.add_argument("--use-first-frame", action="store_true",
                   help="MiniMax H3 continuity chaining for --prompt-dir: "
                        "clip N > 1 uses the last frame of clip N-1 as its "
                        "first-frame input (clip 1 stays pure T2V).")
    p.add_argument("--continuous", action="store_true",
                   help="Chain mode: one image + one prompt -> Qwen-planned "
                        "continuous clips (last frame chains to next)")
    p.add_argument("--image", default=None,
                   help="Reference/start image for --continuous mode")
    p.add_argument("--clips", type=int, default=None,
                   help="Exact clip count for --continuous mode")
    p.add_argument("--duration", type=float, default=None,
                   help="Total target duration in seconds for --continuous")
    p.add_argument("--identity-file", default=None,
                   help="Explicit master identity JSON for --continuous "
                        "(skips automatic identity creation)")
    return p.parse_args(argv)


def run_continuous_command(args, cfg) -> int:
    """ONE IMAGE + ONE PROMPT -> chained continuous video (Mode D).

    Self-contained flow reusing existing pieces only: registry video
    providers via VideoRequest, ComfyUI client, Ollama planner, ffmpeg
    concat/extraction, resolve_dimensions. Never touches storyboard.json,
    state.json, or the image/audio/finalize stages.
    """
    ollama_cfg = cfg.get("ollama", {})
    comfy_cfg = cfg.get("comfyui", {})
    ff_cfg = cfg.get("ffmpeg", {})
    ollama_url = args.ollama_url or ollama_cfg.get("url", "http://127.0.0.1:11434")
    model = getattr(args, "analysis_model", None) or "qwen3:8b"
    comfy_url = args.comfy_url or comfy_cfg.get("url", "http://127.0.0.1:8188")
    project = slugify(args.project) if args.project else slugify(args.prompt[:40])
    base_dir = os.path.join(APP_DIR, "projects", project)
    paths = ch.chain_paths(base_dir)

    source = os.path.abspath(args.image)
    if not os.path.isfile(source):
        print(f"Error: --image not found: {args.image}", file=sys.stderr)
        return 1
    if not st.valid_image_file(source):
        print(f"Error: --image is not a readable image: {args.image}",
              file=sys.stderr)
        return 1
    try:
        target_w, target_h = resolve_dimensions(args.aspect, args.resolution)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        vid_entry = lookup(APP_DIR, "video", args.video_model)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    print(f"Video model: {args.video_model}")
    print(f"Chain planner: {model} (text-only)")
    print(f"Source image: {source}")
    print(f"Target: {target_w}x{target_h}")

    # Storyboard authority: an existing continuous_storyboard.json is the
    # source of truth and is reused verbatim (planner never runs, the CLI
    # prompt is ignored for planning). The CLI prompt is only a storyboard
    # CREATION input used when no board exists yet. This lets users hand-edit
    # the board and re-run any command to generate from it.
    if os.path.isfile(paths["board"]):
        try:
            board = ch.load_board(paths["board"])
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        board_reused = True
        print(f"[Continuous] Existing storyboard found:\n{paths['board']}")
        print("[Continuous] Reusing existing storyboard.")
        print("[Continuous] Skipping planner.")
        print("[Continuous] CLI prompt ignored for storyboard planning.")
        print(f"Loaded continuous storyboard "
              f"({len(board['clips'])} clips) -> {paths['board']}")
        try:
            board = ch.select_clips(board, args.clips)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            ol.check_model_available(ollama_url, model)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        try:
            board = ch.plan_chain(
                args.prompt, model=model, base_url=ollama_url,
                num_clips=args.clips, total_duration=args.duration,
                timeout=int(ollama_cfg.get("timeout", 600)))
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        board["source_image"] = source
        board["planner_model"] = model
        try:
            ch.save_board(board, paths["board"])
        except OSError as exc:
            print(f"Error: cannot save storyboard: {exc}", file=sys.stderr)
            return 1
        print(f"Saved continuous storyboard ({len(board['clips'])} clips) "
              f"-> {paths['board']}")

    if normalize_stop_after(args.stop_after) == "storyboard":
        print("Stopped after storyboard as requested (--stop-after storyboard).")
        return 0

    # Master identity (immutable for the whole chain) + backends/ffmpeg.
    try:
        identity = ch.ensure_identity(
            paths["out_dir"], args.prompt, board, model=model,
            base_url=ollama_url, timeout=int(ollama_cfg.get("timeout", 600)),
            identity_file=getattr(args, "identity_file", None),
            base_dir=base_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Backends + ffmpeg.
    try:
        client = ComfyClient(
            comfy_url,
            timeout=int(comfy_cfg.get("timeout", 60)),
            poll_interval=float(comfy_cfg.get("poll_interval", 2.0)),
            queue_timeout=int(comfy_cfg.get("queue_timeout", 5400)))
        client.check_reachable()
        video_provider = create_provider("video", APP_DIR, args.video_model,
                                         client=client)
        ffmpeg_exe = ff.check_ffmpeg(ff_cfg.get("executable", "ffmpeg"))
    except (RuntimeError, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    motion_clips = None
    if getattr(video_provider, "uses_motion_clips", False):
        if not args.motion_dir:
            print(f"Error: --video-model {args.video_model} requires "
                  f"--motion-dir in --continuous mode.", file=sys.stderr)
            return 1
        try:
            motion_clips = video_provider.discover_motion_clips(args.motion_dir)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Motion dir: {args.motion_dir} ({len(motion_clips)} clips)")

    try:
        ch.run_chain(
            out_dir=paths["out_dir"], board=board, source_image=source,
            provider=video_provider,
            negative_prompt=vid_entry.get("negative_prompt", ""),
            width=target_w, height=target_h, seed=base_seed,
            ffmpeg_exe=ffmpeg_exe, only_clip=args.scene,
            motion_clips=motion_clips, identity=identity)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume the chain.",
              file=sys.stderr)
        return 130
    if args.scene is not None:
        print(f"Clip {args.scene} regenerated. downstream clips start from "
              f"its last frame and may require regeneration; final.mp4 was "
              f"not rebuilt (run without --scene to rebuild it).")
        return 0
    try:
        ch.concat_chain(paths["out_dir"], board["clips"], ffmpeg_exe)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    combo_error = validate_cli_combination(args)
    if combo_error is not None:
        print(f"Error: {combo_error}", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    if bool(getattr(args, "continuous", False)):
        return run_continuous_command(args, cfg)

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
        if args.project:
            project = slugify(args.project)
        elif args.prompt:
            project = slugify(args.prompt[:40])
        else:
            print("Error: --keyframes/--script requires --project", file=sys.stderr)
            return 1
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    try:
        target_w, target_h = resolve_dimensions(args.aspect, args.resolution)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Text-to-video mode: same LTX family, separate backend with its own
    # image-free workflow. The registry key selects the backend; user-facing
    # messages keep the --video-model name.
    is_t2v = getattr(args, "video_mode", "i2v") == "t2v"
    video_model_key = "ltx_t2v" if is_t2v else args.video_model
    # Validate requested models against the registry before doing any work.
    try:
        img_entry = lookup(APP_DIR, "image", args.image_model)
        vid_entry = lookup(APP_DIR, "video", video_model_key)
        aud_entry = lookup(APP_DIR, "audio", args.audio_model)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Motion-control video backends (wan_animate2): driving clips from
    # --motion-dir plus an optional pinned --reference-image reused for
    # every job. Resolved here so planning, resume and stages agree.
    wan_motion = (vid_entry.get("provider") == "wan_animate2"
                  and bool(args.motion_dir))
    if args.motion_dir and vid_entry.get("provider") != "wan_animate2":
        print("Note: --motion-dir is only used by --video-model "
              "wan_animate2; ignoring.")
    reference_image = None
    if args.reference_image:
        if not os.path.isfile(args.reference_image):
            print(f"Error: --reference-image not found: "
                  f"{args.reference_image}", file=sys.stderr)
            return 1
        if vid_entry.get("provider") != "wan_animate2":
            print("Note: --reference-image is only used by --video-model "
                  "wan_animate2; ignoring.")
        else:
            reference_image = os.path.abspath(args.reference_image)

    print(f"Image model: {args.image_model}")
    print(f"Video model: {args.video_model}")
    if is_t2v:
        print("Video mode: t2v (text-to-video, no input image)")
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
    prompt_mode = bool(getattr(args, "prompt_dir", None))
    if prompt_mode:
        # Fail fast on a missing/empty prompt directory before touching
        # ComfyUI, Ollama, or the registry-backed stages.
        prompt_dir_abs = (args.prompt_dir
                          if os.path.isabs(args.prompt_dir)
                          else os.path.join(base_dir, args.prompt_dir))
        try:
            pf.discover_prompt_files(prompt_dir_abs)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if resume_mode:
        if prompt_mode:
            # Prompt-file resume: scenes come from prompt/*.txt verbatim.
            # No storyboard.json is required, read, or written.
            try:
                resume_scenes = prepare_prompt_scenes(
                    args.prompt_dir, base_dir, default_duration, project)
            except (OSError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
        elif not os.path.isdir(base_dir):
            print(f"Error: --resume: project not found: {project}. "
                  f"Generate it first (without --resume).", file=sys.stderr)
            return 1
        else:
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
            only_scene=args.scene,
            video_needs_images=not (wan_motion and reference_image) and not is_t2v
            and vid_entry.get("provider") != "minimax_h3",
            check_video_duration=not wan_motion)
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
        if not resume_mode and not prompt_mode:
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
        video_provider = create_provider("video", APP_DIR, video_model_key, client=client)
        audio_provider = create_provider("audio", APP_DIR, args.audio_model)
    except UnknownModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Motion-control setup: discover driving clips now (fail fast) when a
    # video run is planned; stage-scoped runs (image/audio/finalize) skip it.
    motion_clips: list[str] | None = None
    if getattr(video_provider, "uses_motion_clips", False):
        if "video" in stages_planned:
            if not args.motion_dir:
                print(f"Error: --video-model {args.video_model} requires "
                      f"--motion-dir (directory of 001.mp4, 002.mp4, ...).",
                      file=sys.stderr)
                return 1
            try:
                motion_clips = video_provider.discover_motion_clips(args.motion_dir)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            print(f"Motion dir: {args.motion_dir} ({len(motion_clips)} clips)")
    if reference_image is not None:
        print(f"Reference image: {reference_image}")

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
        "reference_image": reference_image, "motion_clips": motion_clips,
        "motion_dir": args.motion_dir, "prompt_mode": prompt_mode,
        "first_frame_chain": bool(getattr(args, "use_first_frame", False)),
    }

    try:
        if prompt_mode and not resume_mode:
            return run_prompt_fresh_flow(args, ctx, stages_planned)
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
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
