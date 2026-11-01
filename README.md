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

## Opt-in product URL -> short ad

`--auto-ad` is a separate, opt-in entry point. It does not change the normal
prompt/research/storyboard path. It retrieves one HTTP(S) product page,
extracts only verifiable product data, verifies product images, asks local
Ollama to plan a grounded ad, writes Prompt V2 files, and then reuses the
existing image/video/audio/finalize stages.

```powershell
python ai_video/main.py --project potaroma_ad --auto-ad `
  --product-url "https://shop.tiktok.com/us/pdp/..." `
  --video-model minimax_h3_r2v `
  --ad-duration 30 --ad-clips 6 `
  --ad-style hybrid_ugc_cinematic --ad-energy viral `
  --ad-language en --resolution 480 --aspect 9:16 --stop-after all
```

- `--product-url` is required and configurable; there is no hardcoded product.
- `--ad-duration` defaults to `30`; `--ad-language` defaults to `en`.
- `--ad-style` (default `hybrid_ugc_cinematic`) selects the advertising
  format: `hybrid_ugc_cinematic`, `viral_ugc`, `premium_commercial`,
  `storytelling`.
- `--ad-energy` (default `viral`) selects pacing/delivery: `viral`,
  `natural`, `premium`. Style and energy are independent and both are
  passed to Ollama; they change the plan and the generated prompts.
- `--ad-clips` is optional. Without it the clip count is chosen from the
  target duration and the R2V 5-second duration lattice (6 clips -> 31.0s).
- Only `minimax_h3_r2v` is accepted for `--auto-ad`; other video models fail
  clearly instead of silently falling back.
- Product references are separate from character references. No
  `--character-reference-dir` is needed or accepted in auto-ad mode.
- Existing research/plans/prompts/clips/audio are reused only when the
  product URL, product fingerprint, plan fingerprint, ad style/energy and
  files validate. A changed URL, style, energy or plan requires an explicit
  `--overwrite`.
- `--stop-after video` stops before audio and final mux. Use
  `--stop-after all` for a complete spoken ad.
- Auto-ad uses the existing local Piper provider (`piper_en` /
  `en_US-amy-medium`) for English narration by default. The normal pipeline
  still defaults to Edge TTS. Configure `product_ad.audio_model` and
  `product_ad.voice` in `config.yaml` to change this.

Style and energy are stored in `prompt/ad_plan.json` (`ad_style`,
`ad_energy`, `style_rationale`, `energy_rationale`) and in the plan
fingerprint. Plans written before these fields existed used
`ugc_product_showcase`; resuming one with the new defaults is rejected
instead of silently reusing clips — rerun with `--overwrite` to replan.

Planning uses Ollama structured output (a JSON schema) so every clip has
`role`, `shot_type`, `camera`, prompts, spoken text and continuity, and so
`role`/`shot_type` cannot fall outside the allowed values. For R2V chaining,
new plans keep product-only shots contiguous and end on product/macro with a
hybrid bridge before them: a human clip is never placed directly after a
product-only frame, because the chain would have no creator left to preserve
and the model would invent a new person.

Auto-ad project layout:

```
ai_video/projects/<project>/
  source/product.json          # normalized record + provenance
  source/retrieval.json        # fetch/parse provenance
  source/ad_analysis.json      # Ollama marketing analysis
  source/images/               # downloaded raw images
  product_reference/           # verified product identity references
  image_start/1.png            # initial guide frame (first product image)
  prompt/N.txt                 # Prompt V2 video prompts
  prompt/manifest.json         # per-clip durations
  prompt/ad_plan.json          # grounded plan + source/fingerprint metadata
  videos/scene_001.mp4 ...
  audio/scene_001.mp3 ...
  muxed/                       # per-clip muxed video+voiceover
  state.json
  final.mp4
```

Safety and grounding: retrieval validates schemes/hosts, follows redirects
manually, caps download size and rejects private/loopback destinations;
HTML/model output is untrusted data. The planner may cite only extracted
feature IDs. A deterministic claim guard rejects risky marketing terms that
no verified feature supports. Invalid plans are retried a bounded number of
times and never submit a video job.

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
  product_research.py      # opt-in --auto-ad URL research/provenance/images
  ad_planner.py            # grounded Ollama analysis/plan + Prompt V2
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

## Resumable stage workflow

```powershell
# 1. storyboard + images, then stop and inspect
python ai_video/main.py "..." --project mars_city1 --image-model krea --stop-after image
# 2. regenerate only a bad scene (existing storyboard is never touched)
python ai_video/main.py --project mars_city1 --resume --stage image --scene 3 --image-model krea
# 3. continue approved images -> videos (stops after video)
python ai_video/main.py --project mars_city1 --resume --stage video --video-model ltx
# or: resume everything remaining (images -> videos -> audio -> final.mp4)
python ai_video/main.py --project mars_city1 --resume
```

Full staged workflow (`banhmi_world1` example):

```powershell
python ai_video/main.py "..." --project banhmi_world1 --research web --image-model krea --video-model ltx --stop-after image
python ai_video/main.py --project banhmi_world1 --resume --stage video --video-model ltx
python ai_video/main.py --project banhmi_world1 --resume --stage audio
python ai_video/main.py --project banhmi_world1 --resume --stage finalize
```

- `--stop-after storyboard|image(s)|video|audio|all` gates the pipeline (new: `video`, `audio`).
- `--resume` loads `storyboard.json`, re-validates existing outputs (exists/readable/size/format/duration) and skips valid scenes; never calls research/storyboard/Ollama.
- `--stage image|video|audio|finalize` (requires `--resume`) runs one stage; `--scene N` limits to one scene and forces regeneration (not applicable to `finalize`, which always covers all scenes in storyboard order).
- `finalize` muxes each scene's clip + narration, concatenates in storyboard order, and writes `final.mp4` atomically (`.tmp` + validate + rename) without re-encoding when stream-copy is safe; the final file must contain both video and audio streams.
- Progress persists in `state.json` (atomic writes); outputs render to `.tmp` then validate then rename, so good outputs are never destroyed. Ctrl+C prints completed scenes + the resume command.
- Seeds: `--seed` wins, else the run's original `base_seed` from `state.json` is reused.

## Research mode (optional)

```powershell
python ai_video/main.py "Humanoid robots & AI breakthroughs" --research web --stop-after storyboard
python ai_video/main.py "..."   # --research none (default): identical to the old offline pipeline
```

- `--research none` (default): no provider instantiated, no network, no credentials. Behavior unchanged.
- `--research web`: `providers/research/web.py` (`WebResearchProvider`) collects keyless sources (Wikipedia API + best-effort DuckDuckGo instant-answer, stdlib only) and appends them as a **separate labeled message** to Ollama (the original prompt is never modified). Source metadata (`title`/`url`/`published`) is saved under `"research"` in `storyboard.json`; old files without it still load.
- Architecture: `providers/research/base.py` (`ResearchProvider`, `ResearchSource`, `ResearchResult`, `ResearchError`). Failures raise loudly — never fabricated facts. Registry kind `research` + `config/models.json` entry; `main.py` only calls `maybe_research()` and forwards the context string.
- Why external, not Ollama-native: probed local Ollama 0.34.3 — `/api/web_search` and `/api/web_fetch` return **404**, and the DuckDuckGo HTML endpoint is bot-filtered, so HTML scraping is not viable. No API key exists anywhere in this setup; **none is required** (and none is stored). If a keyed search API is ever needed, add a provider that reads e.g. `BRAVE_API_KEY` from the environment.
- Limitation: keyless sources are encyclopedic, not live news — freshness-sensitive topics get background facts, not this week's headlines.

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
