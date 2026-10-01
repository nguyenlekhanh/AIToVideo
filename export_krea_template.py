"""One-time exporter: flatten workflows_template/image_krea2_turbo_t2i.json
(UI format with one Krea-2 Turbo T2I subgraph) into workflows/image/krea2_turbo_t2i.json
(flat ComfyUI API format).

The export is EXACT: every model/LoRA/sampler/sigma/prompt-enhance value comes
from the template. Widget->input correspondence below was read from the
template's node definitions and verified by ComfyUI's own /prompt validation.

Usage:
    python export_krea_template.py     # writes workflows/image/krea2_turbo_t2i.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows_template", "image_krea2_turbo_t2i.json")
DST = os.path.join(HERE, "workflows", "image", "krea2_turbo_t2i.json")
OFF = 1000  # inner subgraph node ids are remapped to id + OFF


def _wid(node):
    return node.get("widgets_values") or []


# Per-node transcription: input_name -> ('L', inner_link_id) for links,
# ('W', widget_index) for literal values from the template.
NODES: dict[int, tuple[str, dict]] = {
    3: ("KSampler", {"model": ("L", 30), "seed": ("L", 77),
                     "steps": ("W", 2), "cfg": ("W", 3),
                     "sampler_name": ("W", 4), "scheduler": ("W", 5),
                     "positive": ("L", 4), "negative": ("L", 14),
                     "latent_image": ("L", 2), "denoise": ("W", 6)}),
    5: ("EmptyLatentImage", {"width": ("L", 75), "height": ("L", 76),
                             "batch_size": ("W", 2)}),
    6: ("CLIPTextEncode", {"text": ("L", 72), "clip": ("L", 10)}),
    8: ("VAEDecode", {"samples": ("L", 7), "vae": ("L", 12)}),
    10: ("UNETLoader", {"unet_name": ("L", 82), "weight_dtype": ("W", 1)}),
    11: ("CLIPLoader", {"clip_name": ("L", 83), "type": ("W", 1), "device": ("W", 2)}),
    12: ("VAELoader", {"vae_name": ("L", 84)}),
    13: ("ConditioningZeroOut", {"conditioning": ("L", 13)}),
    15: ("LoraLoaderModelOnly", {"model": ("L", 16), "lora_name": ("L", 79),
                                 "strength_model": ("L", 80)}),
    16: ("TextGenerate", {"clip": ("L", 18), "prompt": ("L", 19),
                          "max_length": ("L", 74), "sampling_mode": ("W", 2),
                          "sampling_mode.temperature": ("W", 3),
                          "sampling_mode.top_k": ("W", 4),
                          "sampling_mode.top_p": ("W", 5),
                          "sampling_mode.min_p": ("W", 6),
                          "sampling_mode.repetition_penalty": ("W", 7),
                          "sampling_mode.seed": ("W", 8),
                          "thinking": ("L", 73),
                          "use_default_template": ("W", 11)}),
    17: ("StringConcatenate", {"string_a": ("L", 21), "string_b": ("L", 22),
                               "delimiter": ("W", 2)}),
    18: ("PrimitiveStringMultiline", {"value": ("W", 0)}),
    19: ("PrimitiveStringMultiline", {"value": ("L", 45)}),
    20: ("PreviewAny", {"source": ("L", 27)}),
    21: ("ComfySwitchNode", {"switch": ("L", 32), "on_false": ("L", 34),
                             "on_true": ("L", 33)}),
    22: ("ComfySwitchNode", {"switch": ("L", 31), "on_false": ("L", 28),
                             "on_true": ("L", 29)}),
    23: ("PrimitiveBoolean", {"value": ("L", 78)}),
    24: ("PrimitiveBoolean", {"value": ("L", 46)}),
    27: ("StringConcatenate", {"string_a": ("L", 41), "string_b": ("L", 81),
                               "delimiter": ("W", 2)}),
    28: ("ComfySwitchNode", {"switch": ("L", 42), "on_false": ("L", 37),
                             "on_true": ("L", 38)}),
}


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        tpl = json.load(f)
    sub = tpl["definitions"]["subgraphs"][0]
    inner_by_id = {n["id"]: n for n in sub["nodes"]}

    links: dict[int, tuple] = {}
    for L in sub["links"]:
        links[L["id"]] = (L["origin_id"], L["origin_slot"], L["target_id"], L["target_slot"])

    # Outer node 30 widget values feed these subgraph inputs (by input slot).
    outer30 = next(n for n in tpl["nodes"] if n["type"].startswith("b0e5ca93"))
    ow = outer30["widgets_values"]
    subgraph_src: dict[int, tuple] = {
        0: ("val", ow[0]),    # user prompt
        1: ("val", ow[1]),    # prompt_enhance
        2: ("val", ow[2]),    # LLM thinking mode
        3: ("val", ow[3]),    # LLM max tokens
        4: ("node", 49, 0),   # width <- ResolutionSelector
        5: ("node", 49, 1),   # height <- ResolutionSelector
        6: ("val", ow[6]),    # seed
        7: ("val", ow[7]),    # enable_lora
        8: ("val", ow[8]),    # lora_name
        9: ("val", ow[9]),    # strength_model
        10: ("val", ow[10]),  # lora_trigger_word
        11: ("val", ow[11]),  # unet_name
        12: ("val", ow[12]),  # clip_name
        13: ("val", ow[13]),  # vae_name
    }

    def ref(link_id: int):
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
        prompt[str(nid + OFF)] = {"class_type": class_type, "inputs": api_inputs}

    save_node = next(n for n in tpl["nodes"] if n["type"] == "SaveImage")
    res_node = next(n for n in tpl["nodes"] if n["type"] == "ResolutionSelector")
    rw = _wid(res_node)
    prompt["29"] = {"class_type": "SaveImage",
                    "inputs": {"images": [str(8 + OFF), 0],
                               "filename_prefix": _wid(save_node)[0]}}
    prompt["49"] = {"class_type": "ResolutionSelector",
                    "inputs": {"aspect_ratio": rw[0], "megapixels": rw[1],
                               "multiple": rw[2]}}
    return prompt


def main() -> int:
    prompt = build()
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)
    print(f"Wrote {DST} ({len(prompt)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
