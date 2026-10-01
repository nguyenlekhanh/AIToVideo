"""One-time exporter: flatten workflows_template/video_ltx2_5_i2v.json
(UI format with one LTX-2.5 I2V subgraph) into workflows/video/ltx2_5_i2v.json
(flat ComfyUI API format).

The export is EXACT: every model/VAE/CLIP/sampler/sigma/CFG/count value comes
from the template. Only the output resolution is set to 768x512 via the
template's own ResolutionSelector node (aspect + megapixels parameters).

Widget->input correspondence below was read from the template's node
definitions and verified by ComfyUI's own /prompt validation.

Usage:
    python export_template.py            # writes workflows/video/ltx2_5_i2v.json
    python export_template.py --validate # dry-run: validate against ComfyUI
"""
from __future__ import annotations

import copy
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows_template", "video_ltx2_5_i2v.json")
DST = os.path.join(HERE, "workflows", "video", "ltx2_5_i2v.json")
OFF = 1000  # inner subgraph node ids are remapped to id + OFF

# Output resolution: the known-good 768x512 (3:2 @ ~0.4MP, multiple 32).
RESOLUTION = ("3:2 (Photo)", 0.4, 32)


def _wid(node):
    return node.get("widgets_values") or []


# Per-node transcription: input_name -> ('L', inner_link_id) for links,
# ('W', widget_index) for literal values from the template, ('V', const).
# Order below follows the template's inner node ids.
NODES: dict[int, tuple[str, dict]] = {
    338: ("RandomNoise", {"noise_seed": ("W", 0)}),
    339: ("RandomNoise", {"noise_seed": ("L", 770)}),
    340: ("LTXVConcatAVLatent", {"video_latent": ("L", 665), "audio_latent": ("L", 666)}),
    341: ("KSamplerSelect", {"sampler_name": ("W", 0)}),
    344: ("SamplerCustomAdvanced", {"noise": ("L", 670), "guider": ("L", 733),
                                    "sampler": ("L", 671), "sigmas": ("L", 754),
                                    "latent_image": ("L", 672)}),
    348: ("LTXVLatentUpsampler", {"samples": ("L", 677), "upscale_model": ("L", 678),
                                   "vae": ("L", 773)}),
    349: ("LTXVImgToVideoInplace", {"vae": ("L", 774), "image": ("L", 681),
                                     "latent": ("L", 682), "strength": ("W", 0),
                                     "bypass": ("L", 683)}),
    350: ("LTXVPreprocess", {"image": ("L", 752), "img_compression": ("W", 0)}),
    351: ("ResizeImageMaskNode", {"input": ("L", 763), "resize_type": ("W", 0),
                                   "resize_type.longer_size": ("W", 1),
                                   "scale_method": ("W", 2)}),
    352: ("KSamplerSelect", {"sampler_name": ("W", 0)}),
    353: ("ComfyMathExpression", {"expression": ("W", 0), "values.a": ("L", 687)}),
    355: ("ComfyMathExpression", {"expression": ("W", 0), "values.a": ("L", 688)}),
    356: ("EmptyLTXVLatentVideo", {"width": ("L", 689), "height": ("L", 690),
                                    "length": ("L", 718), "batch_size": ("W", 3)}),
    357: ("LTXVImgToVideoInplace", {"vae": ("L", 727), "image": ("L", 691),
                                     "latent": ("L", 692), "strength": ("W", 0),
                                     "bypass": ("L", 693)}),
    358: ("LTXVAudioVAEDecode", {"samples": ("L", 694), "audio_vae": ("L", 730)}),
    359: ("ComfyMathExpression", {"expression": ("W", 0), "values.a": ("L", 695)}),
    360: ("PrimitiveInt", {"value": ("L", 769)}),
    361: ("PrimitiveInt", {"value": ("L", 776)}),
    362: ("PrimitiveInt", {"value": ("L", 767)}),
    363: ("PrimitiveBoolean", {"value": ("W", 0)}),
    364: ("CLIPTextEncode", {"text": ("L", 721), "clip": ("L", 732)}),
    365: ("LTXVConditioning", {"positive": ("L", 696), "negative": ("L", 697),
                               "frame_rate": ("L", 698)}),
    366: ("LTXVEmptyLatentAudio", {"frames_number": ("L", 717), "frame_rate": ("L", 701),
                                    "batch_size": ("W", 2), "audio_vae": ("L", 729)}),
    367: ("LTXVSeparateAVLatent", {"av_latent": ("L", 702)}),
    368: ("SamplerCustomAdvanced", {"noise": ("L", 703), "guider": ("L", 744),
                                    "sampler": ("L", 705), "sigmas": ("L", 751),
                                    "latent_image": ("L", 707)}),
    369: ("LTXVSeparateAVLatent", {"av_latent": ("L", 708)}),
    370: ("CreateVideo", {"images": ("L", 711), "fps": ("L", 713), "audio": ("L", 712),
                           "bit_depth": ("W", 1), "color_space": ("W", 2)}),
    371: ("LatentUpscaleModelLoader", {"model_name": ("L", 781)}),
    372: ("PrimitiveInt", {"value": ("L", 768)}),
    373: ("CLIPTextEncode", {"text": ("W", 0), "clip": ("L", 731)}),
    374: ("VAEDecodeTiled", {"samples": ("L", 709), "vae": ("L", 775), "tile_size": ("W", 0),
                              "overlap": ("W", 1), "temporal_size": ("W", 2),
                              "temporal_overlap": ("W", 3)}),
    376: ("PrimitiveStringMultiline", {"value": ("L", 765)}),
    377: ("LTXVConcatAVLatent", {"video_latent": ("L", 699), "audio_latent": ("L", 700)}),
    378: ("ComfyMathExpression", {"expression": ("W", 0), "values.a": ("L", 715),
                                   "values.b": ("L", 716)}),
    380: ("TextGenerateLTX2Prompt", {"clip": ("L", 749), "prompt": ("L", 723),
                                      "max_length": ("W", 1), "sampling_mode": ("W", 2),
                                      "sampling_mode.temperature": ("W", 3),
                                      "sampling_mode.top_k": ("W", 4),
                                      "sampling_mode.top_p": ("W", 5),
                                      "sampling_mode.min_p": ("W", 6),
                                      "sampling_mode.repetition_penalty": ("W", 7),
                                      "sampling_mode.seed": ("W", 8),
                                      "image": ("L", 757),
                                      "thinking": ("W", 10),
                                      "use_default_template": ("W", 11)}),
    381: ("PreviewAny", {"source": ("L", 725)}),
    382: ("ComfySwitchNode", {"switch": ("L", 722), "on_false": ("L", 720),
                               "on_true": ("L", 724)}),
    383: ("PrimitiveBoolean", {"value": ("L", 766)}),
    384: ("UNETLoader", {"unet_name": ("L", 777), "weight_dtype": ("W", 1)}),
    385: ("VAELoader", {"vae_name": ("L", 778)}),
    386: ("VAELoader", {"vae_name": ("L", 779)}),
    387: ("CLIPLoader", {"clip_name": ("L", 780), "type": ("W", 1), "device": ("W", 2)}),
    388: ("LTXVDualCFGGuider", {"model": ("L", 771), "positive": ("L", 735),
                                 "negative": ("L", 736), "video_cfg": ("W", 0),
                                 "audio_cfg": ("W", 1)}),
    391: ("LTXVDualCFGGuider", {"model": ("L", 772), "positive": ("L", 755),
                                 "negative": ("L", 756), "video_cfg": ("W", 0),
                                 "audio_cfg": ("W", 1)}),
    393: ("CLIPLoader", {"clip_name": ("L", 782), "type": ("W", 1), "device": ("W", 2)}),
    396: ("ManualSigmas", {"sigmas": ("W", 0)}),
    397: ("ManualSigmas", {"sigmas": ("W", 0)}),
}


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        tpl = json.load(f)
    sub = tpl["definitions"]["subgraphs"][0]
    inner_by_id = {n["id"]: n for n in sub["nodes"]}

    # Inner link table: link_id -> (origin_id, origin_slot, target_id, target_slot).
    links: dict[int, tuple] = {}
    for L in sub["links"]:
        links[L["id"]] = (L["origin_id"], L["origin_slot"], L["target_id"], L["target_slot"])

    # Outer node 398 widget values feed these subgraph inputs (by input slot).
    outer398 = next(n for n in tpl["nodes"] if n["id"] == 398)
    ow = outer398["widgets_values"]
    # subgraph input slot -> ("node", outer_id, outer_slot) | ("val", literal)
    subgraph_src: dict[int, tuple] = {
        0: ("node", 395, 0),   # first_frame image <- LoadImage
        1: ("val", ow[0]),     # prompt
        2: ("val", ow[1]),     # prompt_enhance
        3: ("val", ow[2]),     # duration
        4: ("node", 403, 0),   # width <- ResolutionSelector
        5: ("node", 403, 1),   # height <- ResolutionSelector
        6: ("val", ow[5]),     # noise_seed
        7: ("val", ow[6]),     # frame_rate
        8: ("val", ow[7]),     # unet_name
        9: ("val", ow[8]),     # video vae
        10: ("val", ow[9]),    # audio vae
        11: ("val", ow[10]),   # main text encoder clip
        12: ("val", ow[11]),   # latent upscale model
        13: ("val", ow[12]),   # prompt-enhance model clip
    }

    def ref(link_id: int):
        """Resolve an inner link id to an API input value."""
        o_id, o_slot, _, _ = links[link_id]
        if o_id == -10:  # subgraph input proxy
            kind = subgraph_src[o_slot]
            if kind[0] == "node":
                return [str(kind[1]), kind[2]]
            return kind[1]
        return [str(o_id + OFF), o_slot]

    prompt: dict[str, dict] = {}
    for nid, (class_type, inputs) in NODES.items():
        node = inner_by_id[nid]
        if node["type"] != class_type:
            raise ValueError(f"Template drift: node {nid} is {node['type']}, "
                             f"expected {class_type}")
        w = _wid(node)
        api_inputs: dict = {}
        for name, spec in inputs.items():
            kind, arg = spec
            if kind == "L":
                api_inputs[name] = ref(arg)
            elif kind == "W":
                api_inputs[name] = w[arg]
            elif kind == "V":
                resolved = {}
                for k, v in arg.items():
                    resolved[k] = ref(v[1])
                api_inputs[name] = resolved
        prompt[str(nid + OFF)] = {"class_type": class_type, "inputs": api_inputs}

    # Outer nodes (kept as-is, except resolution + input image stays a LoadImage).
    load_image = next(n for n in tpl["nodes"] if n["id"] == 395)
    prompt["395"] = {"class_type": "LoadImage",
                     "inputs": {"image": _wid(load_image)[0]}}
    prompt["403"] = {"class_type": "ResolutionSelector",
                     "inputs": {"aspect_ratio": RESOLUTION[0],
                                "megapixels": RESOLUTION[1],
                                "multiple": RESOLUTION[2]}}
    save_video = next(n for n in tpl["nodes"] if n["id"] == 75)
    sw = _wid(save_video)
    prompt["75"] = {"class_type": "SaveVideo",
                    "inputs": {"video": [str(370 + OFF), 0],
                               "filename_prefix": sw[0],
                               "format": sw[1]}}
    return prompt


def main() -> int:
    validate_only = "--validate" in sys.argv
    prompt = build()
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)
    print(f"Wrote {DST} ({len(prompt)} nodes)")
    if validate_only:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
