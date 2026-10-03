"""One-time exporter: flatten the Wan Animate2 distilled motion-control
template (workflows_template/video_wan_animate2_distilled.json, ComfyUI UI
format with a "Motion Transfer (Wan Animate 2 Distilled)" subgraph) into
workflows/video/wan_animate2.json (flat ComfyUI API format).

Kept live path only (instance 261): reference LoadImage -> cleanGpuUsed ->
subgraph -> frames -> CreateVideo (+pose audio/fps) -> SaveVideo.
Dropped: second dead subgraph instance 603 (no outgoing links), Video
Stitch comparison branch (291/292), BatchImages (261 output wired straight
into CreateVideo), preview/math/notes nodes.

Widget->input correspondence was read from the template plus ComfyUI's own
/object_info input order and verified by ComfyUI's /prompt validation.

Usage:
    python export_wan_animate2_template.py            # writes workflows/video/wan_animate2.json
    python export_wan_animate2_template.py --validate # also POSTs a test prompt (immediately cancelled)
"""
from __future__ import annotations

import copy
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows_template", "video_wan_animate2_distilled.json")
DST = os.path.join(HERE, "workflows", "video", "wan_animate2.json")
OFF = 1000  # inner subgraph node ids are remapped to id + OFF
SUBGRAPH_ID = "11706f8a-428d-4ef9-b24f-863f651c1b0b"  # Motion Transfer instance 261
INSTANCE_ID = 261  # live subgraph instance (603 is a dead branch: no outgoing links)

# Outer slots of instance 261 fed by other outer nodes (slot -> (node, out_slot)).
INSTANCE_LINKS = {0: ("606", 0), 1: ("240", 0), 4: ("605", 0), 6: ("604", 0)}
# Subgraph input slot 2 (continue_motion) is unlinked with no widget: omit it,
# the server default (start fresh) applies.


def _wid(node):
    return node.get("widgets_values") or []


def nid(i: int) -> str:
    return str(i + OFF)


# Inner transcription: input_name -> ("L", link_id) inner link,
# ("S", subgraph_slot), ("W", widget_index), ("V", const).
INNER: dict[int, tuple[str, dict]] = {
    239: ("UNETLoader", {"unet_name": ("S", 18), "weight_dtype": ("W", 1)}),
    9: ("CLIPLoader", {"clip_name": ("S", 19), "type": ("W", 1), "device": ("W", 2)}),
    4: ("CLIPTextEncode", {"text": ("W", 0), "clip": ("L", 8)}),
    3: ("CLIPTextEncode", {"text": ("S", 4), "clip": ("L", 7)}),
    75: ("CLIPVisionLoader", {"clip_name": ("S", 20)}),
    7: ("VAELoader", {"vae_name": ("S", 21)}),
    222: ("CLIPTextEncode", {"text": ("S", 6), "clip": ("L", 590)}),
    257: ("ContextWindowsManual", {
        "model": ("L", 864), "context_length": ("W", 0),
        "context_overlap": ("W", 1), "context_schedule": ("W", 2),
        "context_stride": ("W", 3), "closed_loop": ("W", 4),
        "fuse_method": ("W", 5), "dim": ("W", 6), "freenoise": ("W", 7),
        "cond_retain_index_list": ("W", 8),
        "split_conds_to_windows": ("W", 9),
        "latent_retain_index_list": ("W", 10),
        "causal_window_fix": ("W", 11)}),
    247: ("WanAnimate2ToVideo", {
        "positive": ("L", 669), "negative": ("L", 670), "vae": ("L", 671),
        "width": ("L", 691), "height": ("L", 692), "length": ("S", 17),
        "batch_size": ("W", 3), "video_frame_offset": ("S", 3),
        "pose_strength": ("S", 7), "pose_start_percent": ("S", 8),
        "pose_end_percent": ("S", 9),
        "reference_image_strength": ("S", 5),
        "reference_image": ("L", 672), "pose_video": ("L", 688),
        "clip_vision_output": ("L", 683), "positive_pose": ("L", 674),
        "clip_vision_output_pose": ("L", 675)}),
    258: ("ComfySwitchNode", {"switch": ("S", 10), "on_false": ("L", 865),
                              "on_true": ("L", 740)}),
    76: ("CLIPVisionEncode", {"clip_vision": ("L", 196), "image": ("L", 658),
                              "crop": ("W", 0)}),
    244: ("ResizeImageMaskNode", {"input": ("S", 0)}),  # dynamic combo below
    18: ("BasicScheduler", {"model": ("L", 597), "scheduler": ("W", 0),
                            "steps": ("W", 1), "denoise": ("W", 2)}),
    95: ("ModelSamplingSD3", {"model": ("L", 596), "shift": ("W", 0)}),
    27: ("KSamplerSelect", {"sampler_name": ("W", 0)}),
    224: ("WanAnimate2Cache", {"model": ("L", 700), "device": ("S", 11),
                               "dtype": ("S", 12)}),
    241: ("GetVideoComponents", {"video": ("S", 1)}),
    256: ("GetImageSize", {"image": ("L", 693)}),
    19: ("SamplerCustom", {"model": ("L", 564), "add_noise": ("W", 0),
                           "noise_seed": ("S", 16), "cfg": ("W", 3),
                           "positive": ("L", 679), "negative": ("L", 680),
                           "sampler": ("L", 487), "sigmas": ("L", 486),
                           "latent_image": ("L", 681)}),
    220: ("CLIPVisionEncode", {"clip_vision": ("L", 584), "image": ("L", 641),
                               "crop": ("W", 0)}),
    236: ("ImageFromBatch", {"image": ("L", 661), "batch_index": ("W", 0),
                             "length": ("W", 1)}),
    243: ("ResizeImageMaskNode", {"input": ("L", 655)}),  # dynamic combo below
    6: ("VAEDecode", {"samples": ("L", 594), "vae": ("L", 664)}),
    223: ("TrimVideoLatent", {"samples": ("L", 593), "trim_amount": ("L", 682)}),
    297: ("PrimitiveBoolean", {"value": ("S", 13)}),
    298: ("ImageFromBatch", {"image": ("L", 751), "batch_index": ("W", 0),
                             "length": ("W", 1)}),
    299: ("ComfySwitchNode", {"switch": ("L", 749), "on_false": ("L", 754),
                              "on_true": ("L", 753)}),
}

# ResizeImageMaskNode uses a dynamic combo; API form verified against live
# ComfyUI /prompt: dotted sub-keys plus the combo key itself.
RESIZE_INPUTS = {"resize_type": ("V", "scale dimensions"),
                 "resize_type.crop": ("V", "center"),
                 "scale_method": ("V", "area")}


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
    # Instance widgets cover subgraph slots 3..21 in order (slots 0/1 are
    # link-only, slot 2 has no widget).
    if len(iw) != 19:
        raise ValueError(f"Template drift: instance widgets: {len(iw)}.")

    def sval(slot: int):
        if slot in INSTANCE_LINKS:
            node, out = INSTANCE_LINKS[slot]
            return [node, out]
        if slot == 2:
            return None  # omit: server default
        return iw[slot - 3]

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
        if node_id in (243, 244):
            api_inputs.update({k: v[1] for k, v in RESIZE_INPUTS.items()})
        if node_id == 244:
            api_inputs["resize_type.width"] = ref(694)
            api_inputs["resize_type.height"] = ref(695)
        if node_id == 243:
            api_inputs["resize_type.width"] = sval(14)
            api_inputs["resize_type.height"] = sval(15)
        prompt[nid(node_id)] = {"class_type": class_type,
                                "inputs": api_inputs}

    # Outer nodes (ids kept). Substitutions happen per job in the provider.
    for oid, class_type in ((189, "LoadImage"), (240, "LoadVideo"),
                            (606, "easy cleanGpuUsed"),
                            (604, "PrimitiveStringMultiline"),
                            (605, "PrimitiveStringMultiline"),
                            (288, "GetVideoComponents"),
                            (245, "CreateVideo"), (246, "SaveVideo")):
        if outer_by_id[oid]["type"] != class_type:
            raise ValueError(f"Template drift: outer node {oid} is "
                             f"{outer_by_id[oid]['type']}.")
    prompt["189"] = {"class_type": "LoadImage", "inputs": {"image": None}}
    prompt["240"] = {"class_type": "LoadVideo", "inputs": {"file": None}}
    prompt["606"] = {"class_type": "easy cleanGpuUsed",
                     "inputs": {"anything": ["189", 0]}}
    prompt["604"] = {"class_type": "PrimitiveStringMultiline",
                     "inputs": {"value": _wid(outer_by_id[604])[0]}}
    prompt["605"] = {"class_type": "PrimitiveStringMultiline",
                     "inputs": {"value": None}}
    prompt["288"] = {"class_type": "GetVideoComponents",
                     "inputs": {"video": ["240", 0]}}
    cw = _wid(outer_by_id[245])
    prompt["245"] = {"class_type": "CreateVideo",
                     "inputs": {"images": [nid(299), 0], "fps": ["288", 2],
                                "audio": ["288", 1], "bit_depth": cw[1],
                                "color_space": cw[2]}}
    sw = _wid(outer_by_id[246])
    prompt["246"] = {"class_type": "SaveVideo",
                     "inputs": {"video": ["245", 0], "filename_prefix": None,
                                "format": sw[1], "codec": sw[2]}}
    return prompt


def validate(prompt: dict) -> None:
    """POST the transcribed graph with template inputs. ComfyUI validates
    synchronously: node_errors come back without burning GPU; a prompt_id
    means valid (cancel it immediately)."""
    with open(SRC, encoding="utf-8") as f:
        tpl = json.load(f)
    outer_by_id = {n["id"]: n for n in tpl["nodes"]}
    check = copy.deepcopy(prompt)
    check["189"]["inputs"]["image"] = "1 Cat.jpeg"
    check["240"]["inputs"]["file"] = "0003.mp4"
    check["605"]["inputs"]["value"] = "validation probe"
    check["246"]["inputs"]["filename_prefix"] = "probe/wan_validate"
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
