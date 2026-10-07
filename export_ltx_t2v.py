"""One-time exporter: derive workflows/video/ltx2_5_t2v.json (LTX-2.5
text-to-video, API format) from workflows/video/ltx2_5_i2v.json.

The T2V graph is the verified I2V graph minus image conditioning:
drop LoadImage + ResizeImageMaskNode + LTXVPreprocess + both
LTXVImgToVideoInplace guides, and rewire the two LTXVConcatAVLatent
video_latent inputs straight onto latents (empty latent for the base
pass, upsampled latent for the x2 pass). Conditioning (text prompts),
both guiders, both samplers, duration math (seconds * 24fps + 1),
ResolutionSelector, CreateVideo and SaveVideo are byte-identical.

Usage:
    python export_ltx_t2v.py            # writes workflows/video/ltx2_5_t2v.json
"""
from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows", "video", "ltx2_5_i2v.json")
DST = os.path.join(HERE, "workflows", "video", "ltx2_5_t2v.json")

# Image-conditioning nodes removed for text-to-video, plus the disabled
# prompt-enhancer branch (its switch is permanently off; leaving it would
# dangle on the dropped Preprocess node and fail ComfyUI validation, and
# its dedicated CLIP would waste VRAM).
DROP_IDS = ("395", "1351", "1350", "1349", "1357", "1380", "1393")
DROP_CLASSES = ("LoadImage", "ResizeImageMaskNode", "LTXVPreprocess",
                "LTXVImgToVideoInplace", "TextGenerateLTX2Prompt",
                "CLIPLoader")
DROP_CLASSES = ("LoadImage", "ResizeImageMaskNode", "LTXVPreprocess",
                "LTXVImgToVideoInplace")

# (concat node, new video_latent source) rewires after the drop.
REWIRES = {
    "1377": "1356",  # base pass: empty latent + empty audio
    "1340": "1348",  # upscale pass: upsampled base latent + audio
}


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        src = json.load(f)

    # Drift checks: the I2V graph must look exactly as expected.
    by_class: dict[str, list[str]] = {}
    for nid, node in src.items():
        by_class.setdefault(node.get("class_type", ""), []).append(nid)
    for nid in DROP_IDS:
        if nid not in src:
            raise ValueError(f"Template drift: node {nid} missing.")
    for nid, cls in (("395", "LoadImage"), ("1351", "ResizeImageMaskNode"),
                     ("1350", "LTXVPreprocess"),
                     ("1349", "LTXVImgToVideoInplace"),
                     ("1357", "LTXVImgToVideoInplace"),
                     ("1380", "TextGenerateLTX2Prompt"),
                     ("1393", "CLIPLoader")):
        if src[nid].get("class_type") != cls:
            raise ValueError(f"Template drift: node {nid} is "
                             f"{src[nid].get('class_type')}, expected {cls}.")
    for nid, lat_src in REWIRES.items():
        inputs = src[nid].get("inputs", {})
        if src[nid].get("class_type") != "LTXVConcatAVLatent":
            raise ValueError(f"Template drift: node {nid} is not "
                             f"LTXVConcatAVLatent.")
        if not isinstance(inputs.get("video_latent"), list):
            raise ValueError(f"Template drift: node {nid} has no linked "
                             f"video_latent.")
        if lat_src not in src:
            raise ValueError(f"Template drift: rewire source {lat_src} "
                             f"missing.")

    wf = copy.deepcopy(src)
    for nid in DROP_IDS:
        del wf[nid]
    for nid, lat_src in REWIRES.items():
        wf[nid]["inputs"]["video_latent"] = [lat_src, 0]
    wf["75"]["inputs"]["filename_prefix"] = "video/LTX-2.5_t2v"
    # The orphaned switch branch (enhancer off) repoints at the raw prompt
    # so no link dangles; the switch itself stays false forever.
    wf["1382"]["inputs"]["on_true"] = ["1376", 0]
    # The default prompt text is always overwritten per job; keep a
    # neutral T2V placeholder so the file never mentions a start image.
    wf["1376"]["inputs"]["value"] = (
        "Cinematic documentary footage. (Replaced per scene at runtime.)")

    # Self-checks: no dangling references, no image conditioning left.
    ids = set(wf)
    classes = {n.get("class_type") for n in wf.values()}
    for text in ("LoadImage", "LTXVPreprocess", "LTXVImgToVideoInplace",
                 "TextGenerateLTX2Prompt", "ResizeImageMaskNode"):
        if text in classes:
            raise ValueError(f"Export error: {text} still present.")
    for nid, node in wf.items():
        for name, value in (node.get("inputs") or {}).items():
            if isinstance(value, list) and str(value[0]) not in ids:
                raise ValueError(f"Export error: node {nid} input {name} "
                                 f"dangles at {value}.")
    for required in ("EmptyLTXVLatentVideo", "LTXVConcatAVLatent",
                     "LTXVSeparateAVLatent", "SamplerCustomAdvanced",
                     "LTXVConditioning", "ResolutionSelector", "SaveVideo"):
        if required not in classes:
            raise ValueError(f"Export error: {required} missing.")
    return wf


def main() -> int:
    wf = build()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    print(f"Wrote {DST} ({len(wf)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
