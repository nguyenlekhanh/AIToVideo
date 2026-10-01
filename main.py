"""CLI pipeline. Model-agnostic: only providers + generic requests live here.

Pipeline: Ollama storyboard -> ImageProvider -> VideoProvider ->
AudioProvider -> FFmpeg final.mp4. No node IDs, model filenames, latent
logic, or sampler settings appear in this file.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ffmpeg as ff
import ollama as ol
import storyboard as sb
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal AI video generator (provider architecture).")
    p.add_argument("prompt", help="Video idea, e.g. 'A lone astronaut discovers an ancient city on Mars'")
    p.add_argument("--project", default=None, help="Project name (default: derived from prompt)")
    p.add_argument("--image-model", default="sd15", help="Image model from config/models.json")
    p.add_argument("--video-model", default="ltx", help="Video model from config/models.json")
    p.add_argument("--audio-model", default="edge_tts", help="Audio model from config/models.json")
    p.add_argument("--aspect", default="16:9", help="Target aspect, e.g. 16:9 or 9:16")
    p.add_argument("--resolution", default=720, help="Target resolution, e.g. 420 or 720")
    p.add_argument("--character-reference", default=None,
                   help="Optional character reference image reused for every scene "
                        "(passed through to the image provider; provider decides support)")
    p.add_argument("--stop-after", default="all",
                   choices=["storyboard", "images", "all"],
                   help="Stop the pipeline after the given stage (default: all)")
    # Infra overrides (backward compatible with the pre-provider CLI).
    p.add_argument("--model", default=None, help="Ollama model (default from config.yaml)")
    p.add_argument("--voice", default=None, help="Edge TTS voice (default from config.yaml)")
    p.add_argument("--config", default=os.path.join(APP_DIR, "config.yaml"))
    p.add_argument("--scenes", type=int, default=None, help="Number of scenes")
    p.add_argument("--comfy-url", default=None, help="ComfyUI URL override")
    p.add_argument("--ollama-url", default=None, help="Ollama URL override")
    p.add_argument("--seed", type=int, default=None, help="Base random seed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
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
    duration = float(pipe_cfg.get("duration", 5))
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
    print(f"Target: {target_w}x{target_h}")

    base_dir = os.path.join(APP_DIR, "projects", project)
    img_dir = os.path.join(base_dir, "images")
    vid_dir = os.path.join(base_dir, "videos")
    aud_dir = os.path.join(base_dir, "audio")
    mux_dir = os.path.join(base_dir, "muxed")
    for d in (img_dir, vid_dir, aud_dir, mux_dir):
        os.makedirs(d, exist_ok=True)
    storyboard_path = os.path.join(base_dir, "storyboard.json")
    final_path = os.path.join(base_dir, "final.mp4")

    try:
        ol.check_reachable(ollama_url)
        client = ComfyClient(
            comfy_url,
            timeout=int(comfy_cfg.get("timeout", 60)),
            poll_interval=float(comfy_cfg.get("poll_interval", 2.0)),
            queue_timeout=int(comfy_cfg.get("queue_timeout", 5400)))
        client.check_reachable()
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

    try:
        print("[1/5] Generating storyboard")
        raw = ol.generate_storyboard(args.prompt, model=ollama_model,
                                     base_url=ollama_url, num_scenes=num_scenes,
                                     timeout=int(ollama_cfg.get("timeout", 180)))
        scenes = sb.validate_storyboard(raw)
        subject = SubjectProfile.from_dict(
            raw.get("subject") if isinstance(raw, dict) else None)
        sb.save_storyboard(scenes, storyboard_path, subject)
        print(f"  Saved {len(scenes)} scenes -> {storyboard_path}")
        if args.stop_after == "storyboard":
            print("Stopped after storyboard as requested (--stop-after storyboard).")
            return 0
        if subject is not None:
            print(f"  Subject profile: {subject.subject_type or 'unspecified type'}")

        char_ref = args.character_reference
        if char_ref is not None:
            char_ref = os.path.abspath(char_ref)
            if not os.path.isfile(char_ref):
                print(f"Error: --character-reference not found: {char_ref}",
                      file=sys.stderr)
                return 1
        print("[2/5] Generating scene images")
        image_paths: list[str] = []
        for i, scene in enumerate(scenes, start=1):
            print(f"  scene {i}/{len(scenes)}: image ...")
            scene_prompt = compose_scene_prompt(
                subject, scene["image_prompt"],
                with_reference=char_ref is not None)
            result = image_provider.generate(ImageRequest(
                prompt=scene_prompt, negative_prompt=img_neg,
                width=target_w, height=target_h, seed=base_seed + i,
                output_path=os.path.join(img_dir, f"scene_{i:03d}.png"),
                reference_image_path=char_ref))
            print(f"    -> {result.path} ({result.width}x{result.height})")
            image_paths.append(str(result.path))
        if args.stop_after == "images":
            print(f"Stopped after images as requested (--stop-after images). "
                  f"{len(image_paths)} scene images in {img_dir}.")
            return 0

        print("[3/5] Generating video clips")
        clip_paths: list[str] = []
        for i, scene in enumerate(scenes, start=1):
            print(f"  scene {i}/{len(scenes)}: video ...")
            result = video_provider.generate(VideoRequest(
                prompt=scene["video_prompt"], negative_prompt=vid_neg,
                input_image=image_paths[i - 1], width=target_w, height=target_h,
                duration=duration, seed=base_seed + 1000 + i,
                output_path=os.path.join(vid_dir, f"scene_{i:03d}.mp4")))
            print(f"    -> {result.path} ({result.width}x{result.height})")
            clip_paths.append(str(result.path))

        print("[4/5] Generating narration")
        audio_paths: list[str] = []
        for i, scene in enumerate(scenes, start=1):
            print(f"  scene {i}/{len(scenes)}: audio ...")
            result = audio_provider.generate(AudioRequest(
                text=scene["narration"],
                output_path=os.path.join(aud_dir, f"scene_{i:03d}.mp3"),
                voice=voice, rate=str(tts_cfg.get("rate", "+0%")),
                pitch=str(tts_cfg.get("pitch", "+0Hz"))))
            audio_paths.append(str(result.path))

        print("[5/5] Rendering final video")
        muxed: list[str] = []

        for i, (clip, audio) in enumerate(zip(clip_paths, audio_paths), start=1):
            out = os.path.join(mux_dir, f"scene_{i:03d}.mp4")

            ff.mux_scene(
                clip,
                audio,
                out,
                executable=ffmpeg_exe,
                copy_video=True,   # không encode lại video
            )

            muxed.append(out)

        ff.concat_scenes(
            muxed,
            final_path,
            executable=ffmpeg_exe,
            copy=True,             # không encode lại khi concat
        )

        print(f"Done -> {final_path}")
        return 0
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
