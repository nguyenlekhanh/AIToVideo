"""One-time exporter: flatten workflows_template/video_minimax_h3_t2v.json
(MiniMax H3 UI format with an "Image to Video (MiniMax H3)" subgraph) into
workflows/video/minimax_h3_t2v.json (flat ComfyUI API format).

T2V only: the subgraph's first_frame/last_frame inputs are unlinked in
the template and have no widgets, so they are omitted (server defaults
apply). Everything else is transcribed exactly: loaders/VAEs, turbo
LoRA switch, scheduler, duration math (seconds -> 17k+5 frame lattice),
resolution selector, CreateVideo (+native audio) and SaveVideo.

Widget->input correspondence was read from the template plus ComfyUI's own
/object_info input order and verified by ComfyUI's /prompt validation.

Usage:
    python export_minimax_h3_template.py            # writes workflows/video/minimax_h3_t2v.json
    python export_minimax_h3_template.py --validate # also POSTs a test prompt (immediately cancelled)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows_template", "video_minimax_h3_t2v.json")
DST = os.path.join(HERE, "workflows", "video", "minimax_h3_t2v.json")
OFF = 1000  # inner subgraph node ids are remapped to id + OFF
SUBGRAPH_ID = "79dd8a95-ce9d-4c14-b264-2162e8bec5ce"  # Image to Video (MiniMax H3)
INSTANCE_ID = 140  # the single live subgraph instance

# Subgraph input slots. 0/1 (first_frame/last_frame) are unlinked with no
# widget: omitted, the server default (pure T2V) applies. All other slots
# resolve to the outer instance widget in slot order (slots 2..14).
FIRST_LAST_SLOTS = (0, 1)


def _wid(node):
    return node.get("widgets_values") or []


def nid(i: int) -> str:
    return str(i + OFF)


# Inner transcription: input_name -> ("L", link_id) inner link,
# ("S", subgraph_slot), ("W", widget_index), ("V", const).
INNER: dict[int, tuple[str, dict]] = {
    119: ("VAELoader", {"vae_name": ("S", 9)}),
    120: ("VAELoader", {"vae_name": ("S", 10)}),
    121: ("VAEDecodeAudio", {"samples": ("L", 226), "vae": ("L", 23)}),
    122: ("VAEDecode", {"samples": ("L", 225), "vae": ("L", 8)}),
    123: ("KSamplerSelect", {"sampler_name": ("W", 0)}),
    124: ("BasicScheduler", {"model": ("L", 234), "scheduler": ("W", 0),
                             # steps is LINKED (inner link 235 <- switch 136
                             # -> 20 when turbo is off); the widget (4) is
                             # only the unconnected default and must NOT be
                             # baked in (doing so silently changes the
                             # canonical 20-step path and its workload).
                             "steps": ("L", 235), "denoise": ("W", 2)}),
    125: ("SamplerCustomAdvanced", {"noise": ("L", 40), "guider": ("L", 12),
                                    "sampler": ("L", 16), "sigmas": ("L", 18),
                                    "latent_image": ("L", 188)}),
    126: ("BasicGuider", {"model": ("L", 241), "conditioning": ("L", 187)}),
    127: ("UNETLoader", {"unet_name": ("S", 7), "weight_dtype": ("W", 1)}),
    128: ("CLIPLoader", {"clip_name": ("S", 8), "type": ("W", 1),
                         "device": ("W", 2)}),
    129: ("RandomNoise", {"noise_seed": ("S", 6)}),
    130: ("CreateVideo", {"images": ("L", 167), "audio": ("L", 166),
                          "fps": ("W", 0), "bit_depth": ("W", 1),
                          "color_space": ("W", 2)}),
    131: ("MiniMaxH3ImageToVideo", {"clip": ("L", 189), "vae": ("L", 190),
                                    "prompt": ("S", 2),
                                    # Template links 246/247 wire the outer
                                    # ResolutionSelector (115) slots 0/1 into
                                    # the instance width/height slots: keep
                                    # that wiring so the selector drives
                                    # output resolution per job.
                                    "width": ("V", ["115", 0]),
                                    "height": ("V", ["115", 1]),
                                    "length": ("L", 199)}),
    132: ("ComfyMathExpression", {"expression": ("W", 0), "values.a": ("L", 205)}),
    133: ("PrimitiveFloat", {"value": ("S", 5)}),
    134: ("LoraLoaderModelOnly", {"model": ("L", 229), "lora_name": ("S", 12),
                                  "strength_model": ("S", 13)}),
    135: ("ComfySwitchNode", {"switch": ("L", 238), "on_false": ("L", 232),
                              "on_true": ("L", 233)}),
    136: ("ComfySwitchNode", {"switch": ("L", 239), "on_false": ("L", 236),
                              "on_true": ("L", 237)}),
    137: ("PrimitiveInt", {"value": ("W", 0)}),
    138: ("PrimitiveInt", {"value": ("S", 14)}),
    139: ("PrimitiveBoolean", {"value": ("S", 11)}),
}


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        tpl = json.load(f)
    sub = next(s for s in tpl["definitions"]["subgraphs"]
               if s["id"] == SUBGRAPH_ID)
    inner_by_id = {n["id"]: n for n in sub["nodes"]}
    outer_by_id = {n["id"]: n for n in tpl["nodes"]}

    links: dict[int, tuple] = {}
    for entry in sub["links"]:
        links[entry["id"]] = (entry["origin_id"], entry["origin_slot"])

    inst = outer_by_id[INSTANCE_ID]
    if inst["type"] != SUBGRAPH_ID:
        raise ValueError(f"Template drift: node {INSTANCE_ID} is {inst['type']}.")
    iw = _wid(inst)
    # Instance widgets cover subgraph slots 2..14 in order (slots 0/1 are
    # link-only image inputs with no widget backing).
    if len(iw) != 13:
        raise ValueError(f"Template drift: instance widgets: {len(iw)}.")

    def sval(slot: int):
        if slot in FIRST_LAST_SLOTS:
            return None  # omit: server default (pure T2V, no frames)
        return iw[slot - 2]

    def ref(link_id: int):
        o_id, o_slot = links[link_id]
        if o_id == -10:
            raise ValueError(f"Link {link_id} is a subgraph input proxy; "
                             f"use ('S', slot) instead.")
        return [nid(o_id), o_slot]

    prompt: dict[str, dict] = {}
    for node_id, (class_type, inputs) in INNER.items():
        node = inner_by_id[node_id]
        if node["type"] != class_type:
            raise ValueError(f"Template drift: inner node {node_id} is "
                             f"{node['type']}, expected {class_type}.")
        w = _wid(node)
        api_inputs: dict = {}
        for name, spec in inputs.items():
            kind, arg = spec
            if kind == "L":
                api_inputs[name] = ref(arg)
            elif kind == "S":
                value = sval(arg)
                if value is not None:
                    api_inputs[name] = value
            elif kind == "W":
                api_inputs[name] = w[arg]
            elif kind == "V":
                api_inputs[name] = arg
        prompt[nid(node_id)] = {"class_type": class_type,
                                "inputs": api_inputs}

    # Outer nodes (ids kept). Substitutions happen per job in the provider.
    for oid, class_type in ((115, "ResolutionSelector"), (92, "SaveVideo")):
        if outer_by_id[oid]["type"] != class_type:
            raise ValueError(f"Template drift: outer node {oid} is "
                             f"{outer_by_id[oid]['type']}.")
    sel = _wid(outer_by_id[115])
    prompt["115"] = {"class_type": "ResolutionSelector",
                     "inputs": {"aspect_ratio": sel[0], "megapixels": sel[1],
                                "multiple": 32}}
    prompt["92"] = {"class_type": "SaveVideo",
                    "inputs": {"video": [nid(130), 0],
                               "filename_prefix": None,
                               "format": "auto", "codec": "auto"}}
    return prompt


def validate(prompt: dict) -> None:
    """POST the transcribed graph with template inputs. ComfyUI validates
    synchronously: node_errors come back without burning GPU; a prompt_id
    means valid (cancel it immediately)."""
    check = copy.deepcopy(prompt)
    check[nid(131)]["inputs"]["prompt"] = "validation probe: river at dawn"
    check[nid(133)]["inputs"]["value"] = 5.0
    check[nid(129)]["inputs"]["noise_seed"] = 1
    check["92"]["inputs"]["filename_prefix"] = "probe/minimax_validate"
    data = json.dumps({"prompt": check}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"VALIDATION FAILED:\n{body[:6000]}")
    print("validate OK:", {k: v for k, v in result.items() if k != "node_errors"},
          "node_errors:", result.get("node_errors"))
    pid = result.get("prompt_id")
    if pid:
        data = json.dumps({"delete": [pid]}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8188/queue", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=15).read()
            print(f"validation job {pid} cancelled")
        except Exception as exc:  # noqa: BLE001
            print(f"cancel note: {exc}")


def main() -> int:
    prompt = build()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)
    print(f"Wrote {DST} ({len(prompt)} nodes)")
    if "--validate" in sys.argv:
        validate(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
