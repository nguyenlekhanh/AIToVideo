# AI Video Generator — Project Handoff Summary

**Canonical location:** `ai_video/docs/summary.md`
**Workspace mirror:** `docs/summary.md` (identical content)
**Repository:** `ai_video/` (its own Git repository; the workspace root is not a Git repo)
**Verified Git state:** HEAD `3553145e6b0abb8ef5370bfe6218743a40e10ac5` — `add ads`; no tracked files modified at the time of this review. `AGENTS.md` and the new `docs/` directory are untracked.
**Review date:** 2026-11-01
**Purpose:** Give a new AI model or developer enough verified context to continue work without reading the prior conversation.

> This document was produced by inspecting the source code, configuration, tests, Git history, and generated project artifacts. Where something could not be verified, it is explicitly marked **NEEDS VERIFICATION** or **UNKNOWN**. It deliberately contains no passwords, API keys, tokens, or other secrets.

---

## 1. Project Overview

`ai_video` is a local-first, Python-orchestrated AI video generation pipeline. It turns one of several inputs into a narrated MP4:

```
text prompt / keyframes / source script / numbered prompt files / product URL
        |
        v
Ollama (local LLM: storyboard, planning, analysis)
        |
        v
ComfyUI (local image and video model workflows)
        |
        v
Audio provider (Edge TTS cloud, or local Piper)
        |
        v
FFmpeg (mux + concatenate)
        |
        v
projects/<project>/final.mp4
```

The design is deliberately minimal:

- Python orchestrates; it performs no model inference itself.
- ComfyUI owns image/video model graphs and GPU inference.
- Ollama owns text/vision reasoning.
- FFmpeg/ffprobe own media probing, muxing, concatenation, frame extraction, and audio fitting.
- Persistence is plain JSON files on disk. There is no database, web server, job queue, container, or authentication layer.
- There are no automatic model downloads. The pipeline uses whatever is already installed in ComfyUI and Ollama.
- The only cloud dependency in the normal path is Edge TTS. The opt-in auto-ad path defaults to local Piper instead.

The most recent major feature is the **opt-in `--auto-ad` product-URL-to-short-ad pipeline**, including independent **ad style** and **ad energy** controls. A real end-to-end render was accepted (see §8).

### Intended users / use cases

- Generate short narrative videos from a text idea.
- Generate a storyboard from source images (`--keyframes`) or a long script (`--script`).
- Generate clips from hand-written numbered prompt files (`--prompt-dir`).
- Generate a chained continuous sequence from one image and one prompt (`--continuous`).
- Research a topic on the web before storyboarding (`--research web`).
- Turn a public product page into a grounded, narrated vertical short ad (`--auto-ad`).

### Non-goals (explicit in the code and README)

- No web UI, REST API server, FastAPI, Celery, Redis, Docker, auth, or cloud database.
- No model inference in Python.
- No automatic model downloading.
- No silent fabrication of product facts, prices, ratings, reviews, or claims.
- No on-screen text, captions, subtitles, logos, or watermarks in auto-ad output.

---

## 2. Technology Stack

### Runtime and tools (verified in the review environment)

| Component | Version / detail |
|---|---|
| Python | 3.14.7 |
| Ollama | 0.34.3, local at `http://127.0.0.1:11434` |
| ComfyUI | 0.34.0, local at `http://127.0.0.1:8188` |
| FFmpeg | 9.0.1 (`ffmpeg` and `ffprobe` on PATH) |
| Piper | 1.2.0, local executable under `ai_video/tools/piper/piper.exe` (git-ignored) |
| pytest | 9.1.1 |
| edge-tts | 7.2.8 |
| PyYAML | 6.0.3 |
| requests | 2.34.2 |
| Pillow | 12.3.0 |

### Python dependencies (`ai_video/requirements.txt`)

```
edge-tts>=6.0
pyyaml>=6.0
requests>=2.31
Pillow>=10.0
```

`pytest` is used for development but is not listed in `requirements.txt`. The test suite is written with `unittest` and is pytest-compatible.

### External services and protocols

| Service | Use | Authentication |
|---|---|---|
| Ollama | Text storyboard/planning, vision verification | None; local HTTP |
| ComfyUI | Image/video generation | None; local HTTP |
| Edge TTS | Optional cloud narration | None |
| Product page HTTP(S) | Auto-ad product research | None |
| arXiv API, Hacker News Algolia, Wikipedia API, DuckDuckGo instant answer | Optional `--research web` | None; keyless |
| Piper | Local offline narration | None |
| FFmpeg/ffprobe | Media processing | None |

No API keys are required by the pipeline. If a keyed search API is added later, it should read a `*_API_KEY` environment variable in a new research provider; no such provider exists today.

### What is deliberately absent

No database, ORM, migrations, web framework, frontend, REST/GraphQL server, message broker, scheduler, Dockerfile, Kubernetes manifests, CI configuration, or cloud storage integration. Persistence and coordination are filesystem-based.

---

## 3. Repository and Project Structure

### Workspace layout

```
D:\AI\sources\proj\5\
  ai_video\                     # Git repository root
    AGENTS.md                   # project-context maintenance instructions
    README.md                   # user-facing usage and architecture
    main.py                     # CLI, mode dispatch, stage orchestration
    ad_planner.py               # auto-ad style/energy, analysis, planning, Prompt V2
    product_research.py         # URL safety, retrieval, parsing, image verification
    ollama.py                   # Ollama HTTP transport and prompt helpers
    storyboard.py               # storyboard validation, grounding, persistence
    script.py                   # script-to-storyboard mode
    continuous.py               # continuous chain mode
    promptfiles.py              # numbered prompt files, manifests, refs
    ffmpeg.py                   # probing, frame extraction, audio fit, mux, concat
    state.py                    # state.json schema, atomic persistence, validation
    subject.py                  # subject profile support
    diag_ff.py                  # small FFmpeg diagnostic helper
    export_*.py                 # workflow export/transcription utilities
    config.yaml                 # runtime configuration
    config/models.json          # model registry
    providers/                  # image/video/audio/research provider implementations
    workflows/                  # exported ComfyUI API graphs (tracked)
    workflows_template/         # canonical source graphs (tracked; do not edit exports)
    tests/                      # unittest/pytest suite
    voices/                     # Piper voice models (tracked)
    tools/                      # Piper binary (git-ignored)
    projects/                   # generated artifacts (git-ignored)
  docs/                         # workspace mirror of this handoff document
  test.wav                      # stray workspace file, not part of the repo
  wfprobe.py                    # stray workspace file, not part of the repo
```

### Repository facts

- The Git repository is `ai_video/.git`, not the workspace root.
- `projects/` is git-ignored. Generated videos, state, prompts, and product records are local artifacts.
- `tools/` is git-ignored (Piper binary).
- `voices/` is tracked (34 files).
- `workflows/` is tracked (11 files).
- `tests/` is tracked (40 files).
- `AGENTS.md` is currently untracked in the repository.
- `docs/` did not exist before this report.
- `opencode.json` is tracked and contains a committed provider API key. **Do not copy the value anywhere. Rotate/remove the key** (see §8.6).

### Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Argument parsing, mode selection, stage ordering, resume guards, auto-ad orchestration, final render metadata |
| `ad_planner.py` | Style/energy profiles, creative briefs, Ollama analysis/planning, plan validation, shot-mix rules, Prompt V2 generation, fingerprints |
| `product_research.py` | URL validation, redirect-safe retrieval, TikTok Shop/JSON-LD/meta parsing, image download, vision verification, product record |
| `ollama.py` | `/api/tags` model checks, `/api/chat` JSON transport, storyboard generation, narration/visual rewrites |
| `storyboard.py` | Storyboard schema validation, grounding checks, visual-text checks, save/load |
| `script.py` | Long-text condensation, subject extraction, storyboard generation from script |
| `continuous.py` | Qwen-planned continuous chain, per-clip manifest, last-frame chaining, resume cascade |
| `promptfiles.py` | Numbered prompt discovery, manifest durations, initial image, character references |
| `ffmpeg.py` | Probe, last-frame extraction, audio fitting, mux, concat, validation helpers |
| `state.py` | `state.json` schema, atomic writes, output validation |
| `providers/registry.py` | Lazy provider registry and factory |
| `providers/comfy.py` | Shared ComfyUI HTTP client |
| `providers/dims.py` | Aspect/resolution parsing, snapping, target dimensions |
| `providers/image/*` | SD1.5, SDXL, Flux, Krea image providers |
| `providers/video/*` | LTX I2V/T2V, Wan, Wan Animate2, MiniMax H3 T2V, MiniMax H3 R2V |
| `providers/audio/*` | Edge TTS and Piper |
| `providers/research/*` | Keyless web research provider and pipeline |

### Provider registry

`providers/registry.py` maps model IDs to lazy-imported classes:

```
IMAGE:    sd15, sdxl, flux, krea
VIDEO:    ltx, ltx_t2v, minimax_h3, minimax_h3_r2v, wan, wan_animate2
AUDIO:    edge_tts, edge, piper, piper_en
RESEARCH: web
```

Adding a model requires three things: a ComfyUI workflow file, a provider class, and one `config/models.json` entry. `main.py` does not change.

### ComfyUI workflows

Tracked exports under `ai_video/workflows/` include:

- `image/sd15.json`, `image/sdxl.json`, `image/flux.json`
- `image/krea2_turbo_t2i.json`, `image/krea2_character_ref.json`
- `video/ltx2_5_i2v.json`, `video/ltx2_5_t2v.json`
- `video/wan_i2v.json`, `video/wan_animate2.json`
- `video/minimax_h3_t2v.json`, `video/minimax_h3_r2v.json`

Canonical source graphs live in `workflows_template/`. The R2V export is derived from:

```
workflows_template/video_minimax_h3_r2v_with_H3_Memory_Optimize_H3_Sparse_Attention.json
```

**Do not edit the exported workflows directly.** Change the template and re-run the corresponding `export_*.py` script. The canonical R2V template SHA-256 was verified as:

```
4343814cf3215acf5891c72b494d8f66022aa546bc789a5d06376023f11b9f5c
```

---

## 4. Feature Status

Status definitions:

- **Implemented + verified**: code exists, tests pass, and an end-to-end artifact or direct inspection confirms behavior.
- **Implemented + unit-tested**: code exists and tests pass, but no fresh end-to-end render was performed in this review.
- **Partial / limited**: works within stated constraints; important gaps remain.
- **Not implemented / out of scope**: intentionally absent.

### 4.1 Pipeline modes

| Mode | Entry point | Status | Notes |
|---|---|---|---|
| Text prompt → storyboard → images → video → audio → final | `python main.py "prompt"` | Implemented + verified | Core pipeline. Uses Ollama storyboard, image provider, video provider, audio provider, FFmpeg. |
| Resume / staged workflow | `--resume`, `--stage`, `--scene`, `--stop-after` | Implemented + verified | Filesystem truth plus `state.json`; validates existing outputs before reuse. |
| Numbered prompt files | `--prompt-dir` | Implemented + unit-tested | Skips storyboard/research; each `N.txt` is one clip prompt; optional `manifest.json`. |
| First-frame chaining for prompt mode | `--use-first-frame` | Implemented + unit-tested | MiniMax H3: clip N > 1 uses the previous clip's last frame. |
| Initial image for prompt mode | `--image-1-dir` | Implemented + unit-tested | `1.png/.jpg/.jpeg/.webp` becomes clip 1's first frame; otherwise T2V/R2V behavior. |
| Keyframes → storyboard | `--keyframes DIR` | Implemented + unit-tested | Qwen-VL analyzes source images; no text prompt. |
| Script → storyboard | `--script FILE` | Implemented + unit-tested | Long text is condensed by Qwen; subject profile persisted. |
| Continuous chain | `--continuous` | Implemented + unit-tested | One image + one prompt; Qwen plans clips; each clip starts from the previous last frame. |
| Web research | `--research web` | Implemented + unit-tested | Keyless sources only; not live news. |
| Product URL → short ad | `--auto-ad` | Implemented + verified end-to-end | See §4.2. |

### 4.2 Auto-ad feature status

| Feature | Status | Evidence / limitation |
|---|---|---|
| Public HTTP(S) product URL validation | Implemented | Rejects non-HTTP(S), credentials, localhost/private/link-local/reserved hosts, overlong URLs. Redirects validated manually. |
| Product page retrieval with byte cap | Implemented | `requests` with manual redirect handling and size limits. |
| TikTok Shop structured data parsing | Implemented | Reads `__MODERN_ROUTER_DATA__`; verified on the Potaroma TikTok Shop page. |
| JSON-LD parsing | Implemented | Generic schema.org Product extraction. |
| OpenGraph/meta fallback | Implemented | Title/description/image fallback. |
| Arbitrary e-commerce sites | Partial | Works only when the page exposes usable structured data or meta tags. **NEEDS VERIFICATION** for Amazon, Shopify variants, and heavily JS-rendered pages. |
| Product image download | Implemented | Downloads candidate images, deduplicates, caps bytes, normalizes to PNG. |
| Vision verification of images | Implemented | Ollama vision model (`qwen3-vl:8b` by default) confirms the image shows the exact product. Default is fail-closed (`require_vision_verification: true`). |
| Grounded marketing analysis | Implemented | Ollama produces audience/problem/benefits/objections/angle; benefits must cite feature IDs. |
| Grounded ad planning | Implemented | Ollama structured output with a JSON schema; bounded retries (6). |
| Claim guard | Implemented | Deterministic risky-term check; unsupported claims reject the plan. |
| Style control | Implemented | Four styles: `hybrid_ugc_cinematic`, `viral_ugc`, `premium_commercial`, `storytelling`. |
| Energy control | Implemented | Three energies: `viral`, `natural`, `premium`. Independent from style. |
| Style/energy in plan + fingerprint | Implemented | Stored as `ad_style`, `ad_energy`, rationales; included in `plan_fingerprint`. |
| Legacy plan readability | Implemented | `ugc_product_showcase` is accepted when loading/validating old plans but is not a CLI choice. |
| Prompt V2 generation | Implemented | Numbered prompts with product identity, guide frame, shot direction, style, energy, audio direction, continuity, restrictions. |
| R2V video generation | Implemented + verified | `minimax_h3_r2v` only. Product refs, character refs, and guide frame use separate template slots. |
| Product references separate from character refs | Implemented | `product_refs` is a distinct `VideoRequest` field; product first, then character; guide frame is separate. |
| Per-clip voice-over | Implemented + verified | Piper `piper_en` / `en_US-amy-medium` by default. |
| Final mux/concat | Implemented + verified | 31.046468 s accepted render. |
| Resume safety | Implemented | Rejects changed URL, style, energy, language, duration, clip count, aspect, resolution, video model, audio model, voice, or missing/invalid artifacts. |
| English-only ads | Partial by design | `--ad-language` accepts only `en`/`en-US`/`en-GB`. |
| Other video models for auto-ad | Not implemented by design | `--auto-ad` requires `minimax_h3_r2v`; no silent fallback. |
| Face-forward UGC creator | Not implemented in accepted render | v2 creator is hands-only. A face-forward variant requires a new plan and render. |
| On-screen text / captions / logos | Not implemented by design | Planner schema forces `on_screen_text` to `""`; prompts forbid text. |
| Background music / SFX | Not implemented | Audio stage is narration only. |
| Multi-language ads | Not implemented | English only. |
| Automatic model download | Not implemented | Uses installed models only. |
| Web UI / REST API / database | Not implemented | Out of scope. |

### 4.3 Provider status

| Provider | Status | Notes |
|---|---|---|
| `sd15` | Implemented + unit-tested | Single-checkpoint ComfyUI graph. |
| `sdxl` | Implemented + unit-tested | Single-checkpoint ComfyUI graph. |
| `flux` | Implemented + unit-tested | Single-checkpoint ComfyUI graph. |
| `krea` | Implemented + unit-tested | Krea-2 Turbo T2I; no negative path; live generation tests are opt-in. |
| `ltx` (LTX-2.5 I2V) | Implemented + unit-tested | Verified graph; substitutes prompt, negative, seed, duration, image, dimensions. |
| `ltx_t2v` | Implemented + unit-tested | Text-to-video variant of LTX-2.5. |
| `minimax_h3` (T2V) | Implemented + unit-tested | Optional first-frame chaining; 17k+5 frame lattice at 24 fps. |
| `minimax_h3_r2v` | Implemented + verified end-to-end | Reference-to-video; product/character refs plus guide frame; memory optimization and sparse attention nodes preserved. |
| `wan` (Wan2.1 I2V) | Implemented + unit-tested | Historical commit says “wan not finish yet”; no fresh end-to-end quality verification in this review. Treat as usable-but-less-proven. |
| `wan_animate2` | Implemented + unit-tested | Motion-control provider; fixed ~5 s output; requires `--motion-dir`; optional pinned `--reference-image`. |
| `edge_tts` / `edge` | Implemented | Cloud TTS. In this environment Edge TTS returned HTTP 403, so auto-ad uses Piper. |
| `piper` | Implemented + unit-tested | Local offline TTS; Vietnamese voices under `voices/piper/vn`; WAV converted to MP3. The Vietnamese path was not re-rendered in this review. |
| `piper_en` | Implemented + verified | English voice `en_US-amy-medium`; used by the accepted auto-ad render. |
| `web` research | Implemented + unit-tested | Keyless arXiv/HN/Wikipedia/DDG; not live news. |

### 4.4 Test suite status

The repository contains 39 `test_*.py` modules (plus `tests/__init__.py`; 40 tracked files under `tests/`). Test counts by file include:

```
test_product_ad.py       76
test_prompt_files.py     44
test_continuous.py       43
test_minimax_h3_r2v.py   40
test_keyframes.py        35
test_script.py           34
test_character_refs.py   33
test_grounding.py        27
test_identity.py         26
test_visual_prompts.py   26
test_minimax_h3.py       25
test_krea.py             22
test_image_1_dir.py      21
test_wan_animate2.py     21
test_piper.py            20
test_ltx_first_frame.py  19
test_ltx_t2v.py          19
test_resume.py           19
test_first_frame.py      17
test_news_gate.py        17
test_research.py         16
test_finalize.py         13
test_registry.py         13
test_semantic.py         13
test_ltx.py              11
test_cli.py              10
test_storyboard.py       10
test_subject.py           9
test_prompt_leak.py       9
test_regeneration.py      9
test_video_duration.py    9
test_state.py             8
test_comfy.py             7
test_research_v2.py       7
test_image.py             6
test_wan.py               5
test_ffmpeg.py            3
test_audio.py             2
test_character.py         1
```

Verified commands and results:

```powershell
cd ai_video
python -m pytest -q tests/test_product_ad.py
# 76 passed, 12 subtests passed in 1.08s

python -m pytest -q -k "not TestKreaLiveValidate and not TestKreaLiveGenerate and not TestCharacterLive and not ImageValidationTest"
# 739 passed, 6 deselected, 143 subtests passed in 58.16s
```

Live/ComfyUI-dependent tests are intentionally excluded from the broad run. There is no CI configuration.

---

## 5. Data, Database, and APIs

### 5.1 Database

There is **no database**. No SQLite, Postgres, Redis, ORM, migrations, or schema versioning beyond JSON `version` fields.

All persistence is local JSON plus media files under `ai_video/projects/<project>/`.

For the style/energy feature cycle there were **no database changes** (there is no database) and **no new application API endpoints**. The feature is confined to planning, validation, prompt generation, plan metadata, and resume fingerprinting. Auto-ad's outbound HTTP calls are the existing Ollama/ComfyUI clients plus product-page retrieval.

### 5.2 Project artifact layout

Normal pipeline project:

```
ai_video/projects/<project>/
  storyboard.json
  state.json
  source.txt                  # --script mode only
  condensed_script.json       # --script mode only
  images/scene_001.png ...
  videos/scene_001.mp4 ...
  audio/scene_001.mp3 ...
  muxed/scene_001.mp4 ...
  final.mp4
```

Auto-ad project:

```
ai_video/projects/<project>/
  source/product.json         # normalized product record + provenance + fingerprint
  source/retrieval.json       # fetch/parse provenance
  source/ad_analysis.json     # Ollama marketing analysis
  source/images/              # downloaded raw images
  product_reference/          # verified product identity references (PNG)
  image_start/1.png           # initial guide frame (first verified product image)
  prompt/N.txt                # Prompt V2 video prompts
  prompt/manifest.json        # per-clip durations
  prompt/ad_plan.json         # grounded plan + product refs + fingerprints
  videos/scene_001.mp4 ...
  continuity/scene_001_last.png ...
  audio/scene_001.mp3 ...
  muxed/scene_001.mp4 ...
  state.json
  final.mp4
```

Continuous-chain project:

```
ai_video/projects/<project>/
  continuous_storyboard.json
  clips/
    manifest.json
    final.mp4
    001/
      start.png
      prompt.txt
      video.mp4
      lastframe.png
    002/ ...
```

### 5.3 `state.json` schema

`state.py` defines `STATE_VERSION = 1`.

```json
{
  "version": 1,
  "project": "project_name",
  "base_seed": 123456789,
  "scene_ids": [1, 2, 3],
  "stages": {
    "research":   {"status": "pending|complete", "attempts": 0},
    "storyboard": {"status": "pending|complete", "attempts": 0},
    "image":      {"status": "pending|complete", "completed_scenes": [], "attempts": {}},
    "video":      {"status": "pending|complete", "completed_scenes": [], "attempts": {}},
    "audio":      {"status": "pending|complete", "completed_scenes": [], "attempts": {}},
    "finalize":   {"status": "pending|complete", "attempts": 0}
  },
  "updated_at": "2026-10-31T00:00:00+00:00"
}
```

Auto-ad adds render metadata under `stages.video.render` and `stages.audio.render`:

```json
{
  "stages": {
    "video": {
      "render": {
        "width": 480,
        "height": 864,
        "aspect": "9:16",
        "resolution": "480",
        "video_model": "minimax_h3_r2v",
        "ad_style": "hybrid_ugc_cinematic",
        "ad_energy": "viral"
      }
    },
    "audio": {
      "render": {"model": "piper_en", "voice": "en_US-amy-medium"}
    }
  }
}
```

Important: `state.json` is advisory. On resume every claimed output is re-validated for existence, readability, non-zero size, format, and plausible duration. Missing or corrupt state starts fresh rather than crashing.

### 5.4 Product record (`source/product.json`)

Verified fields include:

```json
{
  "source_url": "https://...",
  "canonical_url": "https://...",
  "retrieved_at": "ISO-8601",
  "retrieval_status": "complete|partial",
  "source_type": "tiktok_shop_product|...",
  "title": "...",
  "display_name": "...",
  "brand": "...",
  "description": "...",
  "category": "...",
  "price": {"amount": 0, "currency": "USD", "display": "USD 0", "source_id": "src_1"},
  "rating": null,
  "review_count": null,
  "sold_count": null,
  "features": [{"id": "f1", "text": "...", "source_id": "src_1", ...}],
  "image_candidates": [{"id": "img_1", "url": "...", ...}],
  "selected_images": [
    {
      "id": "img_1",
      "path": "absolute or project-relative path",
      "source_url": "...",
      "sha256": "...",
      "width": 800,
      "height": 800,
      "verification": {
        "matches": true,
        "confidence": 0.95,
        "visible_product": "...",
        "reason": "...",
        "model": "qwen3-vl:8b"
      }
    }
  ],
  "sources": [{"id": "src_1", "url": "...", "title": "...", "publisher": "...", "retrieved_at": "...", "source_type": "...", "confidence": "high"}],
  "facts": [{"field": "...", "value": "...", "source_id": "src_1", "feature_id": "f1"}],
  "diagnostics": [],
  "fingerprint": "sha256"
}
```

`product_fingerprint` is a SHA-256 over source/canonical URL, title, brand, description, feature texts, and image source URLs. It is used for research reuse and stale-artifact detection.

### 5.5 Ad plan (`prompt/ad_plan.json`)

`AD_PLAN_VERSION = 1`. Core fields:

```json
{
  "version": 1,
  "ad_title": "...",
  "audience": "...",
  "core_message": "...",
  "style": "hybrid_ugc_cinematic",
  "ad_style": "hybrid_ugc_cinematic",
  "ad_energy": "viral",
  "style_rationale": "...",
  "energy_rationale": "...",
  "language": "en",
  "target_duration_seconds": 30.0,
  "clip_count": 6,
  "clip_duration_seconds": 5.0,
  "actual_duration_seconds": 31.0,
  "clips": [
    {
      "id": 1,
      "role": "hook",
      "shot_type": "human_ugc",
      "camera": "handheld",
      "image_prompt": "...",
      "video_prompt": "...",
      "spoken_text": "...",
      "on_screen_text": null,
      "continuity": "...",
      "feature_ids": ["f1"],
      "product_reference_ids": ["img_1", "img_2", "img_3"]
    }
  ],
  "claims_used": [{"text": "...", "feature_ids": ["f1"]}],
  "claims_avoided": ["..."],
  "cta": "...",
  "audio_plan": {"mode": "per_clip_voiceover", "voice": "en_US-amy-medium"},
  "planner_model": "qwen3:8b",
  "product_fingerprint": "sha256",
  "plan_fingerprint": "sha256",
  "product": {
    "source_url": "...",
    "canonical_url": "...",
    "title": "...",
    "brand": "...",
    "price": null,
    "sources": [],
    "fingerprint": "sha256"
  },
  "product_references": [
    {
      "id": "img_1",
      "path": "...",
      "source_url": "...",
      "sha256": "...",
      "verification": {}
    }
  ]
}
```

Allowed roles: `hook`, `problem`, `feature`, `demo`, `benefit`, `lifestyle`, `cta`.

Allowed shot types: `human_ugc`, `hybrid`, `product_showcase`, `macro_detail`.

`plan_fingerprint` hashes product fingerprint, target/clip durations, clip count, style, ad style, ad energy, language, clips, claims, and CTA. Style and energy are independent inputs; clips generated for one pair must never be reused for another.

### 5.6 Prompt V2 (`prompt/N.txt` and `prompt/manifest.json`)

Each `N.txt` is a complete prompt containing:

- Shot number, vertical 9:16, style, energy header.
- Product identity and brand.
- Product reference tags (`<Picture 1>` ... `<Picture 3>`).
- Guide frame tag (`<Picture N+1>`): clip 1 uses the verified product image; later clips use the previous clip's actual last frame.
- Shot direction and camera.
- Action from the plan.
- Style and energy direction.
- Audio/voice-over direction (spoken separately; no lip-sync).
- Continuity notes.
- Restrictions: no cuts, new subjects, on-screen text, logos, watermarks, price tags, review screenshots, fake interfaces, additional products, or unsupported claims.

`manifest.json`:

```json
{
  "clips": {
    "1": {"duration": 5.0},
    "2": {"duration": 5.0}
  }
}
```

### 5.7 Storyboard (`storyboard.json`)

`storyboard.py` saves:

```json
{
  "research": {"mode": "web", "sources": [...]},
  "subject": {...},
  "scenes": [
    {
      "id": 1,
      "duration": 5,
      "narration": "...",
      "image_prompt": "...",
      "video_prompt": "...",
      "research_fact_ids": [],
      "source_ids": [],
      "grounding": "creative|factual"
    }
  ]
}
```

Validation covers required fields, durations, grounding, visual prompt cleanliness (no text/collage/charts/numbers), and semantic checks.

### 5.8 HTTP endpoints used

#### Ollama (`http://127.0.0.1:11434`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/tags` | Check model availability and reachability |
| POST | `/api/chat` | JSON-mode chat; supports `format` schema, `images` base64, `options.temperature`, `options.num_predict` |

No Ollama-native web search is used. `/api/web_search` and `/api/web_fetch` were probed and return 404 in this environment.

#### ComfyUI (`http://127.0.0.1:8188`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/system_stats` | Reachability check |
| POST | `/prompt` | Queue a workflow; returns `prompt_id` |
| POST | `/queue` | Cancel/remove queued jobs |
| GET | `/queue` | Inspect queue state |
| POST | `/interrupt` | Interrupt current job |
| GET | `/history/{prompt_id}` | Poll job completion and outputs |
| GET | `/view?filename=...` | Download generated output |
| POST | `/upload/image` | Upload reference/guide images |

#### External HTTP

- Product page URL: GET with manual redirect validation and byte cap.
- Edge TTS: outbound cloud synthesis via the `edge-tts` package.
- Web research: arXiv API, Hacker News Algolia API, Wikipedia API, DuckDuckGo instant answer.

There is no application HTTP server, no public API, no authentication, and no CORS layer.

### 5.9 Security model

- Product HTML, JSON-LD, page text, and model output are **untrusted data**, never instructions.
- URL validation rejects non-HTTP(S) schemes, credentials, overlong URLs, localhost/.local/.internal/.home/.lan names, and hosts resolving to private/loopback/link-local/reserved/multicast/unspecified addresses; fragments are dropped during normalization.
- Redirects are followed manually and each target is re-validated.
- Download size is capped (`max_image_bytes`, default 8,000,000).
- Image paths are validated as readable image files before use.
- Product facts must come from extracted features; risky claim terms must be supported by a cited feature.
- Auto-ad prompts explicitly forbid on-screen text, logos, watermarks, price tags, review screenshots, and fake interfaces.

---

## 6. Configuration

### 6.1 `config.yaml`

```yaml
ollama:
  url: "http://127.0.0.1:11434"
  model: "qwen3:8b"
  vision_model: "qwen3-vl:8b"
  num_scenes: 3
  timeout: 180
  planning_timeout: 600

comfyui:
  url: "http://127.0.0.1:8188"
  timeout: 60
  poll_interval: 2.0
  queue_timeout: 5400

pipeline:
  duration: 5

tts:
  voice: "en-US-AriaNeural"
  rate: "+0%"
  pitch: "+0Hz"

ffmpeg:
  executable: "ffmpeg"
  ffprobe_executable: "ffprobe"
  crf: 19
  preset: "veryfast"

product_ad:
  timeout: 30
  vision_timeout: 240
  max_image_bytes: 8000000
  audio_model: "piper_en"
  voice: "en_US-amy-medium"
  require_vision_verification: true
```

Key semantics:

- `ollama.planning_timeout` is used for auto-ad analysis/planning; `ollama.timeout` is used for normal storyboard work.
- `comfyui.queue_timeout` defaults to 5400 s because LTX-2.5 jobs are slow.
- `pipeline.duration` is the fallback scene duration; storyboard scene durations win.
- `product_ad.audio_model` / `voice` override the normal Edge TTS default only in auto-ad mode.
- `product_ad.require_vision_verification: true` is fail-closed. Set false only for offline/testing; the code will otherwise refuse unverified images.

### 6.2 `config/models.json`

Registry entries include provider ID, workflow path, negative prompt, and model-specific defaults:

- Image: `sd15` (20 steps, CFG 7.0), `sdxl` (8 steps, CFG 2.0), `flux` (4 steps, CFG 1.0), `krea` (denoise 0.65, character-ref workflow).
- Video: `ltx`, `wan`, `wan_animate2`, `ltx_t2v`, `minimax_h3`, `minimax_h3_r2v`.
- Audio: `edge_tts`/`edge`, `piper` (Vietnamese voice dir), `piper_en` (English voice dir).
- Research: `web` (max 5 sources, 20 s timeout, 8 facts, relevance threshold 0.15, 2 expansions).

### 6.3 CLI flags

#### Core

| Flag | Default | Meaning |
|---|---|---|
| positional `prompt` | — | Video idea; required unless `--resume`, `--keyframes`, `--script`, or `--prompt-dir` |
| `--project` | derived from prompt | Project slug under `projects/` |
| `--image-model` | `sd15` | Image model key |
| `--video-model` | `ltx` | Video model key |
| `--video-mode` | `i2v` | `i2v` or `t2v` (LTX only) |
| `--audio-model` | `edge_tts` | Audio provider key |
| `--aspect` | `16:9` | Target aspect |
| `--resolution` | `720` | Target resolution |
| `--seed` | random | Base seed; resume reuses stored seed |
| `--scenes` | config `num_scenes` | Scene count |
| `--config` | `ai_video/config.yaml` | Config path |
| `--comfy-url`, `--ollama-url` | config | Service URL overrides |
| `--model` | config `ollama.model` | Ollama text model |
| `--voice` | config `tts.voice` | Edge voice name or Piper voice stem |

#### Stage/resume

| Flag | Meaning |
|---|---|
| `--stop-after storyboard / image / images / video / videos / audio / all` | Stop after a stage |
| `--stage image / video / audio / finalize` | Run one stage; requires `--resume` |
| `--scene N` | Limit to one scene; requires `--resume`; forces regeneration |
| `--resume` | Load existing storyboard/prompts and skip valid outputs |

#### Input modes

| Flag | Meaning |
|---|---|
| `--keyframes DIR` | Source keyframe images analyzed by Qwen-VL |
| `--analysis-model` / `--analysis` | Ollama model for keyframe/script/continuous analysis |
| `--script FILE` | Source text for script-to-storyboard |
| `--subject auto / none / god` | Master subject profile for script mode |
| `--prompt-dir DIR` | Numbered prompt files (`1.txt`, `2.txt`, ...) |
| `--use-first-frame` | MiniMax H3 chaining for prompt mode |
| `--image-1-dir DIR` | Optional initial image for prompt mode |
| `--character-reference FILE` | Single character reference image |
| `--character-reference-dir DIR` | Up to 4 numbered character references; requires prompt mode + R2V |
| `--reference-image FILE` | Pinned reference for Wan Animate2 |
| `--motion-dir DIR` | Driving motion clips for Wan Animate2 |
| `--continuous` | Continuous chain mode |
| `--image FILE` | Start image for continuous mode |
| `--clips N` | Exact clip count for continuous mode |
| `--duration SECONDS` | Total target duration for continuous mode |
| `--identity-file FILE` | Explicit master identity JSON for continuous mode |

#### Auto-ad

| Flag | Default | Meaning |
|---|---|---|
| `--auto-ad` | off | Enable product-URL-to-ad mode |
| `--product-url URL` | — | Required public HTTP(S) product page |
| `--ad-duration SECONDS` | `30.0` | Target ad duration |
| `--ad-clips N` | auto | Optional 1–8 clip count |
| `--ad-style` | `hybrid_ugc_cinematic` | `hybrid_ugc_cinematic`, `viral_ugc`, `premium_commercial`, `storytelling` |
| `--ad-energy` | `viral` | `viral`, `natural`, `premium` |
| `--ad-language` | `en` | English only |
| `--vision-model` | config `ollama.vision_model` | Product image verifier |
| `--overwrite` | off | Explicitly replace auto-ad research/prompts/clips |

Auto-ad restrictions enforced before any work:

- Cannot combine with `--continuous`, a positional prompt, `--prompt-dir`, `--keyframes`, `--script`, `--character-reference`, `--character-reference-dir`, or `--image-1-dir`.
- Requires `--video-model minimax_h3_r2v` and `--video-mode i2v`.
- `--stage image` is rejected.
- `--stop-after` must be `video`, `audio`, or `all`.
- `--scene` requires `--resume`; `--resume` requires `--project`.
- `--ad-duration` must be positive; `--ad-clips` must be 1–8.
- `--overwrite` has no effect outside `--auto-ad`.

### 6.4 Environment variables

None are required. The code does not read a `.env` file. `.gitignore` excludes `.env`, `config/credentials.json`, `config/tokens.json`, keys, certificates, and similar files.

If a keyed research provider is added later, it should read an environment variable such as `BRAVE_API_KEY`; do not hardcode keys.

### 6.5 Recommended auto-ad invocation

```powershell
cd ai_video
python main.py --project potaroma_ad_hybrid_v2 --auto-ad `
  --product-url "https://shop.tiktok.com/us/pdp/..." `
  --video-model minimax_h3_r2v `
  --ad-duration 30 --ad-clips 6 `
  --ad-style hybrid_ugc_cinematic --ad-energy viral `
  --ad-language en --resolution 480 --aspect 9:16 --stop-after all
```

Always pass `--aspect 9:16` for auto-ad. The generated prompts and plan explicitly describe vertical 9:16, but the code does not force the aspect; using the default `16:9` would create a mismatch.

---

## 7. Technical Decisions and History

### 7.1 Git history (verified)

```
3553145 2026-10-31 add ads
80d3b74 2026-10-30 ads
5cb74b5 2026-10-19 image firstframe, image dir, prompt dit
58fb30d 2026-10-09 minimax --use-last-frame
ce3d965 2026-10-08 minimax
eaea390 2026-10-07 add t2v
eb02403 2026-10-05 fix text in image
2dccf22 2026-10-03 wan not finish yet
e1baffc 2026-10-01 add function --keyframe
ea59963 2026-10-01 add vietnam voice piper
be16362 2026-10-01 fix audio
4fb2829 2026-10-01 add final stage
43714eb 2026-09-30 fix prompt leakage
110264b 2026-09-30 Initial commit
```

The style/energy work is present in HEAD `3553145`; no tracked files were modified at review time.

### 7.2 Architecture decisions

1. **Provider architecture.** `main.py` knows only `ImageProvider`, `VideoProvider`, `AudioProvider`, and generic request/result dataclasses. Model-specific node IDs, checkpoints, samplers, latent handling, and dimension constraints live in provider classes. This keeps adding models localized.

2. **Lazy provider imports.** `providers/registry.py` imports provider modules only when instantiated, so registry metadata/listing does not require model dependencies.

3. **Filesystem truth.** `state.json` records progress, but every claimed output is re-validated on resume. Missing/corrupt state starts fresh; it never replaces actual file checks.

4. **Atomic outputs.** Media renders to `.tmp`, validates, then `os.replace`s into place. A failed or interrupted render does not destroy a good prior output.

5. **No model downloads.** The pipeline uses installed ComfyUI/Ollama models. Missing models produce clear errors instead of silent downloads.

6. **Untrusted page/model data.** Product HTML, JSON-LD, and model output are data only. The planner may cite only extracted feature IDs and verified image IDs. A deterministic claim guard rejects risky terms unsupported by a cited feature.

7. **Vision verification is fail-closed by default.** Product images must be confirmed by a vision model unless the operator explicitly disables it.

8. **Product references are distinct from character references.** `VideoRequest.product_refs` and `character_refs` are separate concepts. R2V places product references first, then character references; neither is ever used as the guide/continuity frame.

9. **Guide frame is separate.** The R2V guide-frame slot is used for clip 1's verified product image or the previous clip's actual last frame. It is never treated as a product identity reference.

10. **Single-take ad.** Auto-ad plans are one continuous sequence with no cuts or new scenes. Later clips must continue the same action and keep the product or hand visible.

11. **R2V continuity rule.** A product-only frame contains no creator. A `human_ugc` clip immediately after `product_showcase` or `macro_detail` would force the model to invent a new person. New plans therefore require a hybrid bridge or contiguous product shots. This rule is strict only for new plans; old rendered plans remain loadable.

12. **Style and energy are independent.** Style controls storytelling/format; energy controls pacing/delivery. Both are sent to Ollama, both shape prompts, and both are part of the plan fingerprint. Every style/energy pair must produce a different creative brief and different prompts.

13. **Legacy style compatibility.** `ugc_product_showcase` is readable when validating/loading old plans so resume fails with a precise message instead of an unreadable file, but it is not a CLI choice.

14. **Auto-ad is opt-in and isolated.** It does not change the normal prompt/research/storyboard path. Its flags are inert unless `--auto-ad` is present.

15. **Auto-ad uses local Piper by default.** Edge TTS returned HTTP 403 in this environment. `piper_en` / `en_US-amy-medium` is the default for auto-ad; the normal pipeline still defaults to Edge TTS.

16. **No on-screen text.** The planner schema forces `on_screen_text` to an empty string, and prompts forbid text/captions/logos/watermarks. This avoids garbled generated text and unsupported claims.

17. **Duration lattice.** R2V quantizes 5.0 s to 124 frames (5.1667 s) at 24 fps. Six clips therefore produce 31.0 s, not exactly 30.0 s. The plan records `actual_duration_seconds`; finalize pads narration to the quantized clip duration.

18. **Bounded retries.** Planning retries up to 6 times with the validation error appended, so small local models can correct schema mistakes. Invalid plans never submit a video job.

### 7.3 Style/energy implementation details

`ad_planner.py` defines:

```python
AD_STYLES = ("hybrid_ugc_cinematic", "viral_ugc", "premium_commercial", "storytelling")
AD_ENERGIES = ("viral", "natural", "premium")
DEFAULT_AD_STYLE = "hybrid_ugc_cinematic"
DEFAULT_AD_ENERGY = "viral"
LEGACY_AD_STYLE = "ugc_product_showcase"
LEGACY_AD_ENERGY = "natural"
```

Each style profile contains: summary, planning rules, visual line, dialogue direction, CTA direction, restriction, per-shot visual descriptions, and per-shot human/product balance lines.

Each energy profile contains: summary, planning rules, prompt line, and delivery descriptor.

`_creative_brief(style, energy)` combines both into the Ollama planning prompt. `_build_clip_prompt` folds the style/visual and energy/pacing lines into each Prompt V2 file.

Validation rejects unknown style/energy values before research or rendering. `plan_config_problem` compares a reused plan's target duration, style, energy, language, and clip count against the current request.

### 7.4 Shot-mix enforcement

`_enforce_shot_mix` requires, for new plans with 3+ clips:

- `hybrid_ugc_cinematic`: at least one human UGC shot and at least one product showcase/macro shot.
- `viral_ugc`: at least one creator-led human UGC shot.
- `premium_commercial`: at least one product showcase/macro shot; at most half the clips may be human-led.
- `storytelling`: opens with hook/problem, includes a demo/feature/benefit/lifestyle clip, ends with CTA.

For non-premium styles, strict continuity rejects `product_showcase|macro_detail` → `human_ugc` transitions.

### 7.5 Auto-ad execution flow

1. Validate CLI combination and style/energy.
2. Resolve project name (explicit `--project`, else derived from URL).
3. Reject a project that already has a standard `storyboard.json` or unmanaged numbered prompts, unless `--overwrite`.
4. If an existing `prompt/ad_plan.json` exists and `--overwrite` is absent:
   - Load plan and product record.
   - Require the same normalized product URL.
   - Require matching target duration, style, energy, language, and clip count.
   - Validate all selected product images.
   - Rewrite/validate Prompt V2 without overwriting differing files.
5. Otherwise:
   - Check Ollama text and vision models are available.
   - Retrieve and verify product data.
   - Analyze the product with Ollama.
   - Plan the ad with structured output and bounded retries.
   - Write Prompt V2, manifest, and `ad_plan.json`.
6. Build scenes from numbered prompts with narration from the plan.
7. Require every scene to have spoken text.
8. Load/reuse `state.json`, preserving `base_seed` unless `--seed` is given.
9. Determine stages from `--stage`, `--overwrite`, or resume planning.
10. Apply `--stop-after` gating.
11. Build render/audio metadata and reject resume if inputs changed.
12. Prepare the initial guide frame from the first verified product image.
13. Run video (R2V), audio (Piper), finalize (fit audio to quantized duration, mux, concat).
14. Write `final.mp4` atomically and validate both video and audio streams.

### 7.6 Resume rejection rules

`auto_ad_resume_problem` rejects reuse when:

- Stored video render metadata is missing and no valid first clip can establish dimensions.
- Width, height, aspect, resolution, or video model differ.
- Stored audio render metadata is missing.
- Audio model or voice differs.
- The first clip's actual dimensions do not match the requested render.

Additionally, plan reuse rejects changed product URL, target duration, style, energy, language, or clip count, and missing/invalid product images.

### 7.7 Historical issue that shaped the continuity rule

The first accepted-style render, `potaroma_ad_hybrid`, used this shot order:

```
human_ugc, product_showcase, human_ugc, macro_detail, human_ugc, human_ugc
```

It had product→human transitions at clips 2→3 and 4→5. Because each R2V clip starts from the previous clip's last frame, those transitions had no creator in the source frame, and the model invented a second creator. This is the **v1 continuity defect**. The project is kept as evidence.

The v2 plan used a safe order:

```
human_ugc, human_ugc, human_ugc, hybrid, product_showcase, macro_detail
```

The hybrid bridge keeps the creator present before the product-only block. v2 was accepted.

---

## 8. Current Status

### 8.1 Feature status summary

- The normal pipeline is functional and well covered by unit tests.
- The auto-ad pipeline is complete for its stated scope: English, `minimax_h3_r2v`, local Piper, vertical ads, product-URL research, grounded planning, Prompt V2, resume safety.
- Style and energy controls are fully implemented, validated, persisted, fingerprinted, and tested.
- A real v2 auto-ad render was accepted.
- The v1 auto-ad render is retained as evidence of the continuity defect and should not be used as a quality reference.

### 8.2 Accepted v2 render (verified with ffprobe)

Project: `ai_video/projects/potaroma_ad_hybrid_v2`

| Property | Value |
|---|---|
| Duration | 31.046468 s |
| Dimensions | 480 × 864 (vertical 9:16) |
| Frame rate | 24 fps |
| Frames | 744 |
| Video codec | h264 |
| Audio codec | aac |
| Size | 4,417,635 bytes |
| Clips | 6 × 5 s (quantized to 5.1667 s each) |
| Style | `hybrid_ugc_cinematic` |
| Energy | `viral` |
| Planner | `qwen3:8b` |
| Vision verifier | `qwen3-vl:8b` |
| Audio | `piper_en` / `en_US-amy-medium` |
| Plan fingerprint | `4abea2c0ea66ac5a95bca12378ebada13ad928d7e95ab686a3f9ff44bf4c505f` |
| Product fingerprint | `8f30ddfd05880251673e7b27f7ebf894d608dd4e384a8527976daf73d3d295bf` |

Shot order:

```
1 hook          human_ugc
2 lifestyle     human_ugc
3 feature       human_ugc
4 demo          hybrid
5 benefit       product_showcase
6 cta           macro_detail
```

Observed quality: consistent creator (hands-only), consistent cat, consistent product, no second creator, no on-screen text, CTA delivered in voice-over.

### 8.3 v1 render (rejected quality reference)

Project: `ai_video/projects/potaroma_ad_hybrid`

- Same duration/dimensions/codecs as v2 (31.046468 s, 480×864, 744 frames).
- Shot order: `human_ugc, product_showcase, human_ugc, macro_detail, human_ugc, human_ugc`.
- Continuity defect at 2→3 and 4→5: a new creator was invented after product-only frames.
- Kept as evidence; do not resume or cite as a successful render.

### 8.4 Test status

- Focused auto-ad suite: **76 passed, 12 subtests passed**.
- Broader non-live suite: **739 passed, 6 deselected, 143 subtests passed**.
- No CI pipeline; tests are run manually.

### 8.5 Known issues and limitations

1. **Cross-project absolute image paths.** `potaroma_ad_hybrid` and `potaroma_ad_hybrid_v2` product records point `selected_images` at `projects/potaroma_ad/product_reference/*`. Those projects therefore depend on `potaroma_ad` files continuing to exist. Deleting `potaroma_ad` breaks resume/reuse for the hybrid projects. This is a real coupling bug/limitation; a future fix should copy verified images into each project or store project-relative paths.

2. **R2V pre-flight dimension log mismatch.** The R2V pre-flight logs a `480x832` estimate while the actual output is `480x864`. Resume uses real file dimensions, so this is informational/logging noise, not a render failure. **NEEDS VERIFICATION** if exact dimension reporting becomes important.

3. **Edge TTS HTTP 403 in this environment.** Auto-ad defaults to Piper. The normal pipeline still defaults to Edge TTS; if Edge is blocked, pass `--audio-model piper_en --voice en_US-amy-medium`.

4. **v2 creator is hands-only.** There is no face-forward UGC creator in the accepted render. A face-forward variant requires a new plan and a new render; do not assume the current plan supports it.

5. **Auto-ad aspect is not forced.** Prompts say vertical 9:16, but the CLI accepts other aspects. Always pass `--aspect 9:16`; otherwise prompts and output dimensions can disagree.

6. **Auto-ad is English-only and R2V-only.** Other languages and video models are rejected by design.

7. **Product page support is not universal.** TikTok Shop structured data, JSON-LD, and meta fallbacks work. Arbitrary JS-heavy or anti-bot pages may fail. The code fails loudly rather than fabricating data.

8. **No automated end-to-end auto-ad test.** The 76 tests are unit-level with mocks/fakes. A real render requires Ollama, ComfyUI, Piper, and FFmpeg.

9. **No background music/SFX.** Only voice-over narration is generated.

10. **No on-screen text/captions.** Intentional.

11. **`projects/` and `tools/` are git-ignored.** Artifacts and the Piper binary are not in Git. A fresh clone needs the Piper binary and voices (voices are tracked; the binary is not).

12. **No CI.** Tests are manual. Live ComfyUI tests are skipped unless the service is running.

13. **`opencode.json` contains a committed provider API key.** This is a security issue. Do not copy the value into docs or logs. Rotate the key and remove it from the repository; prefer an environment variable or an untracked local config. The file is tracked in Git history, so rotation is required even if the file is later cleaned.

14. **Legacy `ugc_product_showcase` plans.** They are readable but not runnable with the new default style. Resume rejects them with a precise message; rerun with `--overwrite` to replan.

15. **`--overwrite` is destructive by design.** It replaces auto-ad research, prompts, and generated clips. Use deliberately.

16. **Plan validation is conservative.** Risky claim terms require exact supporting feature text. Valid creative copy may be rejected if the product record lacks the corresponding feature wording.

17. **Wan2.1 status.** The code and tests exist, but the historical commit message says “wan not finish yet.” No fresh end-to-end quality verification was performed in this review. Treat it as implemented-but-less-proven.

18. **`AGENTS.md` is untracked.** It exists in `ai_video/` but is not committed. Decide whether to track it.

### 8.6 Security / secrets

- No secrets are included in this document.
- `ai_video/opencode.json` is tracked and contains a provider API key. The value was deliberately not copied. Rotate/remove it.
- `.gitignore` covers `.env`, credentials, tokens, keys, certificates, logs, databases, generated media, `projects/`, and `tools/`.
- Product page content and model output are treated as untrusted. Prompt-injection defenses are explicit in `product_research.py` and `ad_planner.py`.

### 8.7 Development priorities / next steps

Recommended, in order:

1. **Rotate and remove the committed provider key** from `opencode.json`; purge it from Git history if appropriate.
2. **Fix cross-project product image paths.** Copy verified images into each project's `product_reference/` and store project-relative paths so projects are self-contained.
3. **Add an automated or scripted end-to-end smoke test** for auto-ad using a tiny duration/clip count, with clear skips when services are unavailable.
4. **Decide the face-forward UGC variant.** If desired, add a new plan/style variant and verify identity consistency across the R2V chain.
5. **Enforce or warn on `--aspect 9:16` for auto-ad** to prevent prompt/output mismatch.
6. **Improve R2V dimension reporting** so pre-flight logs match actual output.
7. **Consider tracking `AGENTS.md`** and documenting the handoff workflow in the repo.
8. **Add CI** for the non-live test subset.
9. **Optionally add background music/SFX** as a separate audio layer if product requirements call for it.
10. **Optionally add a keyed research provider** for live news; do not pretend keyless sources are live news.

---

## 9. Important Files

### Entry points and orchestration

| File | Why it matters |
|---|---|
| `ai_video/main.py` | CLI, mode dispatch, stage orchestration, auto-ad flow, resume guards, render metadata. ~107 KB. |
| `ai_video/ad_planner.py` | Style/energy profiles, analysis, planning, validation, shot mix, Prompt V2, fingerprints. ~69 KB. |
| `ai_video/product_research.py` | URL safety, retrieval, parsing, image download/verification, product record. ~45 KB. |
| `ai_video/ollama.py` | Ollama transport and storyboard/rewrite calls. ~27 KB. |
| `ai_video/storyboard.py` | Storyboard validation/grounding/persistence. ~16 KB. |
| `ai_video/script.py` | Script-to-storyboard mode. ~21 KB. |
| `ai_video/continuous.py` | Continuous chain mode. ~56 KB. |
| `ai_video/promptfiles.py` | Numbered prompts, manifests, initial images, character refs. ~13 KB. |
| `ai_video/ffmpeg.py` | Probe, last-frame extraction, audio fit, mux, concat. ~9 KB. |
| `ai_video/state.py` | State schema, atomic persistence, output validation. ~7 KB. |

### Providers

| File | Why it matters |
|---|---|
| `providers/registry.py` | Model registry and lazy factory. |
| `providers/comfy.py` | Shared ComfyUI HTTP client. |
| `providers/dims.py` | Aspect/resolution resolution and snapping. |
| `providers/video/minimax_h3_r2v.py` | R2V workflow injection, product/character refs, guide frame. |
| `providers/video/ltx.py`, `ltx_t2v.py` | LTX-2.5 I2V/T2V. |
| `providers/video/minimax_h3.py` | MiniMax H3 T2V with optional first-frame chaining. |
| `providers/video/wan.py`, `wan_animate2.py` | Wan I2V and motion-control. |
| `providers/image/*` | SD1.5, SDXL, Flux, Krea. |
| `providers/audio/edge_tts.py`, `piper_tts.py` | Cloud and local narration. |
| `providers/research/web.py` | Keyless web research. |

### Configuration and workflows

| File | Why it matters |
|---|---|
| `config.yaml` | Runtime settings, including `product_ad`. |
| `config/models.json` | Model registry entries and defaults. |
| `requirements.txt` | Runtime Python dependencies. |
| `workflows/video/minimax_h3_r2v.json` | Exported R2V graph used at runtime. |
| `workflows_template/video_minimax_h3_r2v_with_H3_Memory_Optimize_H3_Sparse_Attention.json` | Canonical R2V template. Do not edit. |
| `export_minimax_h3_r2v.py` | Re-exports the R2V graph from the template. |
| `workflows/` | Other exported ComfyUI graphs. |

### Tests and docs

| File | Why it matters |
|---|---|
| `tests/test_product_ad.py` | 76 focused auto-ad tests. |
| `tests/test_cli.py` | CLI combination validation. |
| `tests/test_resume.py` | Resume behavior. |
| `tests/test_state.py` | State persistence/validation. |
| `tests/test_minimax_h3_r2v.py` | R2V injection/reference behavior. |
| `tests/test_prompt_files.py` | Prompt file/manifest behavior. |
| `README.md` | User-facing usage and architecture. |
| `AGENTS.md` | Instructions to maintain this handoff document. |
| `docs/summary.md` | This document. |

### Generated reference projects

| Path | Why it matters |
|---|---|
| `projects/potaroma_ad_hybrid_v2/` | Accepted v2 render; use as the quality/reference artifact. |
| `projects/potaroma_ad_hybrid/` | v1 render with the continuity defect; evidence only. |
| `projects/potaroma_ad/` | Original product research and reference images; hybrid projects currently depend on its `product_reference/` files. |

---

## 10. AI Handoff Instructions

### 10.1 Read order

1. `ai_video/AGENTS.md` — rules for maintaining this document.
2. `ai_video/docs/summary.md` — this handoff.
3. `ai_video/README.md` — user-facing usage.
4. `ai_video/main.py` — actual orchestration and CLI behavior.
5. `ai_video/ad_planner.py` and `ai_video/product_research.py` — auto-ad core.
6. Relevant provider modules and tests for the area you will change.

### 10.2 Ground rules

- **Do not document planned functionality as implemented.** Keep status labels honest.
- **Do not include secrets.** Never copy values from `opencode.json`, `.env`, or credential files.
- **Do not edit exported workflows directly.** Edit the template and re-export.
- **Do not change `AD_PLAN_VERSION` or fingerprint semantics casually.** Old plans and resume behavior depend on them.
- **Preserve legacy plan readability.** Old `ugc_product_showcase` plans must still fail with a clear message, not crash.
- **Keep style and energy independent.** Do not collapse them into one dimension.
- **Keep strict continuity for new plans only.** Old rendered plans must remain loadable.
- **Keep product refs, character refs, and guide frame separate.** Do not merge these concepts.
- **Keep auto-ad opt-in.** Normal pipeline behavior must not change when `--auto-ad` is absent.
- **Respect untrusted data.** Product HTML and model output are never instructions.
- **Use atomic writes and validate before replacing outputs.**
- **Do not delete `projects/` artifacts without explicit user approval.** They are evidence and reference renders.

### 10.3 How to run the project

```powershell
cd ai_video

# Normal pipeline
python main.py "A lone astronaut discovers an ancient city on Mars" `
  --project mars_city --image-model flux --video-model ltx `
  --aspect 16:9 --resolution 720

# Auto-ad
python main.py --project my_ad --auto-ad `
  --product-url "https://shop.tiktok.com/us/pdp/..." `
  --video-model minimax_h3_r2v `
  --ad-duration 30 --ad-clips 6 `
  --ad-style hybrid_ugc_cinematic --ad-energy viral `
  --aspect 9:16 --resolution 480 --stop-after all
```

Prerequisites:

- Ollama running at the configured URL with `qwen3:8b` and `qwen3-vl:8b` installed (or configured alternatives).
- ComfyUI running at the configured URL with the required workflows/models.
- `ffmpeg` and `ffprobe` on PATH.
- Piper binary at `tools/piper/piper.exe` and the configured voice under `voices/piper/en/` for auto-ad.

### 10.4 How to verify changes

```powershell
cd ai_video

# Focused auto-ad tests
python -m pytest -q tests/test_product_ad.py

# Broader non-live suite (skips ComfyUI-dependent live tests)
python -m pytest -q -k "not TestKreaLiveValidate and not TestKreaLiveGenerate and not TestCharacterLive and not ImageValidationTest"

# Optional: unittest discovery
python -m unittest discover -s tests
```

Before claiming an end-to-end success, verify the actual MP4:

```powershell
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,codec_type,width,height,nb_frames -of json projects\<project>\final.mp4
```

Check for both a video and an audio stream. For auto-ad, confirm dimensions, duration, and the absence of a second creator / on-screen text by inspecting frames or watching the video.

### 10.5 How to add a model/provider

1. Add the ComfyUI workflow under `workflows/<kind>/`.
2. Add a provider class implementing the appropriate base interface.
3. Register the provider in `providers/registry.py`.
4. Add a `config/models.json` entry.
5. Add tests for graph substitution scope, dimensions, and error handling.
6. Do not add model-specific logic to `main.py`.

### 10.6 How to change auto-ad planning

- Style/energy profiles live in `STYLE_PROFILES` / `ENERGY_PROFILES` in `ad_planner.py`.
- Planning prompt text lives in `PLAN_SYSTEM` and `_creative_brief`.
- Schema constraints live in `PLAN_SCHEMA`.
- Validation and claim guards live in `validate_plan` / `_enforce_shot_mix` / `_unsupported_claim_term`.
- Prompt V2 text lives in `_build_clip_prompt`.
- Fingerprints live in `plan_fingerprint`.
- If you add a style or energy, update the CLI choices, tests, README, and this document.

### 10.7 How to safely resume or regenerate

- Re-running the same auto-ad command resumes completed stages.
- Use `--stop-after video` to inspect clips before audio/finalize.
- Use `--stage video --scene N --resume` to regenerate one clip; downstream clips may need regeneration and `final.mp4` is not rebuilt.
- Use `--overwrite` only when you intend to replace research, prompts, and generated clips.
- If a plan/style/energy changed, resume is rejected; replan with `--overwrite`.

### 10.8 Known traps

- `--auto-ad` requires `--video-model minimax_h3_r2v` explicitly. The default `ltx` will fail validation.
- `--auto-ad` requires `--aspect 9:16` in practice, even though the CLI does not enforce it.
- `--auto-ad` rejects projects that already contain a standard `storyboard.json`.
- `--auto-ad` rejects `--character-reference*`, `--image-1-dir`, `--prompt-dir`, `--keyframes`, and `--script`.
- The v2 creator is hands-only; do not promise a face-forward creator without a new plan/render.
- The hybrid projects' product images currently point into `potaroma_ad`; do not delete that project.
- Edge TTS may return 403; use Piper for auto-ad.
- `projects/` and `tools/` are git-ignored; a fresh clone is not a complete runtime environment.
- `opencode.json` contains a committed key; rotate it.

### 10.9 Suggested next work

If continuing development, the highest-value fixes are:

1. Rotate/remove the committed provider key.
2. Make auto-ad projects self-contained by copying verified product images and using project-relative paths.
3. Add a scripted end-to-end smoke test for auto-ad.
4. Enforce or warn on `--aspect 9:16` in auto-ad.
5. Add CI for the non-live test subset.
6. Decide whether a face-forward UGC variant is required.
7. Improve R2V pre-flight dimension reporting.

---

## Appendix A — Verified commands and outputs

```text
git -C ai_video log -1 --format='%H %s'
3553145e6b0abb8ef5370bfe6218743a40e10ac5 add ads

git -C ai_video status --short
?? AGENTS.md
?? docs/          # after this report was written

python -m pytest -q tests/test_product_ad.py
76 passed, 12 subtests passed in 1.08s

python -m pytest -q -k "not TestKreaLiveValidate and not TestKreaLiveGenerate and not TestCharacterLive and not ImageValidationTest"
739 passed, 6 deselected, 143 subtests passed in 58.16s
```

ffprobe on the accepted v2 render:

```json
{
  "streams": [
    {"codec_name": "h264", "codec_type": "video", "width": 480, "height": 864, "r_frame_rate": "24/1", "nb_frames": "744"},
    {"codec_name": "aac", "codec_type": "audio", "r_frame_rate": "0/0", "nb_frames": "678"}
  ],
  "format": {"duration": "31.046468", "size": "4417635", "bit_rate": "1138328"}
}
```

## Appendix B — Status legend

| Label | Meaning |
|---|---|
| Implemented + verified | Code, tests, and an end-to-end artifact or direct inspection confirm it. |
| Implemented + unit-tested | Code and tests exist; no fresh end-to-end render in this review. |
| Partial / limited | Works within stated constraints; meaningful gaps remain. |
| Not implemented / out of scope | Intentionally absent. |
| NEEDS VERIFICATION | Not confirmed by source inspection or execution in this review. |
| UNKNOWN | No reliable information available. |

## Appendix C — Document maintenance

Per `ai_video/AGENTS.md`, update this document when:

- A completed feature changes.
- An ongoing/unfinished feature changes.
- Architecture changes.
- Database or API behavior changes.
- A new technical decision is made or an old one is reversed.
- A bug is discovered or resolved.
- Development priorities or next steps change.
- Important files or dependencies are added/removed.

Keep it consistent with the source code, do not document planned work as implemented, do not include secrets, and preserve historical decisions that explain why the code looks the way it does.
