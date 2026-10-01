# AI Video Generator (provider architecture)

Minimal pipeline. Python orchestrates; specialist tools do the work:

```
prompt -> Ollama (storyboard JSON) -> ImageProvider -> VideoProvider
       -> AudioProvider (Edge TTS) -> FFmpeg final.mp4
```

No database, web UI, FastAPI, Celery, Redis, Docker, auth, or cloud APIs
(except Edge TTS). No model inference in Python. No automatic model
downloads — uses whatever is already in `ComfyUI/models/`.

## Setup

```powershell
pip install -r ai_video/requirements.txt
```

Requires running locally:

- Ollama at `http://127.0.0.1:11434` (model configurable)
- ComfyUI at `http://127.0.0.1:8188` (URL configurable)
- `ffmpeg` on PATH

## Usage

```powershell
python ai_video/main.py "A lone astronaut discovers an ancient city on Mars" --project mars_city --image-model flux --video-model ltx --aspect 16:9 --resolution 720
python ai_video/main.py "..." --image-model sdxl --video-model wan --aspect 9:16 --resolution 720
python ai_video/main.py "..." --project mars_city   # defaults: sd15 + ltx, 16:9, 720
```

Old-style invocations still work (`--model`, `--voice`, `--project`, `--seed`).

Startup prints a summary (`Image/Video model`, `Aspect`, `Resolution`,
`ComfyUI`, `Target`) plus per-scene final dimensions reported by each provider.

## Output

```
ai_video/projects/<project_name>/
  storyboard.json
  images/scene_001.png ...
  videos/scene_001.mp4 ...
  audio/scene_001.mp3 ...
  final.mp4
```

## Architecture

`main.py` knows only provider interfaces (`ImageProvider`, `VideoProvider`,
`AudioProvider`) plus generic request/response dataclasses. All model logic
(node IDs, checkpoints, samplers, latent handling, dimension constraints)
lives in providers:

```
ai_video/
  main.py                  # pipeline only, no model logic
  config/
    models.json            # registry: model -> provider + workflow + defaults
  providers/
    comfy.py               # shared ComfyUI HTTP client (queue/wait/download)
    errors.py              # ProviderError / UnknownModelError
    dims.py                # aspect/resolution -> target dims, snapping
    registry.py            # lookup + factory (no pipeline changes to add models)
    image/base.py          # ImageProvider / ImageRequest / ImageResult
    image/comfyui.py       # shared single-checkpoint graph provider
    image/sd15.py image/sdxl.py image/flux.py   # thin per-model subclasses
    image/krea.py          # Krea-2 Turbo provider (own graph, T2I only, no negative path)
    video/base.py          # VideoProvider / VideoRequest / VideoResult
    video/ltx.py           # LTX-2.5 provider (verified graph, 5 inputs + dims)
    video/wan.py           # Wan2.1 provider
    audio/base.py audio/edge_tts.py
  workflows/
    image/sd15.json image/sdxl.json image/flux.json
    image/krea2_turbo_t2i.json     # exact export of workflows_template (do not edit)
    video/ltx2_5_i2v.json  # exact export of workflows_template (do not edit)
    video/wan_i2v.json
  workflows_template/video_ltx2_5_i2v.json  # LTX-2.5 source (export via export_template.py)
  tests/                   # unittest suite (run from ai_video/)
```

Adding a model = workflow file + provider class + registry entry.

Resolution: `--aspect`/`--resolution` resolve to target dims app-wide
(720 = frame height landscape / frame width portrait, snapped to 32);
each provider adjusts to its model constraints and reports final dims.

## Tests

```powershell
cd ai_video
python -m unittest discover -s tests
```

Unit tests cover registry, dims, CLI parsing, substitution scope, and error
wrapping. Validation tests submit each provider graph to ComfyUI `/prompt`
(queued then cancelled — needs ComfyUI, skipped otherwise). End-to-end runs
of `main.py` produce the final proof MP4.
