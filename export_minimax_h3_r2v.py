"""One-time exporter: flatten workflows_template/...r2v... (UI format,
already flat API-style nodes with links as arrays) into
workflows/video/minimax_h3_r2v.json (flat ComfyUI API format).

The template is already flat (no subgraph), so this exporter just
transcribes UI node/widget/link structure into the API format used by
the rest of the project ({"node_id": {"class_type":..., "inputs":{...}}}).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "workflows_template",
                   "video_minimax_h3_r2v_with_H3_Memory_Optimize_H3_Sparse_Attention.json")
DST = os.path.join(HERE, "workflows", "video", "minimax_h3_r2v.json")


def build() -> dict:
    with open(SRC, encoding="utf-8") as f:
        tpl = json.load(f)
    nodes = {str(n["id"]): n for n in tpl["nodes"]}
    # link id -> (origin_id, origin_slot)
    linkmap = {}
    for L in tpl["links"]:
        # [id, origin_id, origin_slot, target_id, target_slot, type]
        linkmap[L[0]] = (str(L[1]), L[2])
    prompt = {}
    for nid, n in nodes.items():
        if n.get("type") in ("MarkdownNote",):
            continue
        class_type = n["type"]
        inputs = {}
        # linked inputs
        for inp in n.get("inputs", []) or []:
            name = inp["name"]
            link = inp.get("link")
            if link is not None:
                o_id, o_slot = linkmap[link]
                inputs[name] = [o_id, o_slot]
        # widget inputs: from widgets_values_named if present, else positional
        named = n.get("widgets_values_named")
        if named:
            # widgets_values_named keys are the WIDGET names (scheduler, steps,
            # denoise), but the input slot "steps" is LINKED (link 289), so
            # only unlinked widget inputs should be filled from named values.
            for inp in (n.get("inputs") or []):
                w = inp.get("widget")
                if not w:
                    continue
                if inp["name"] in inputs:
                    continue  # linked
                wn = w.get("name")
                if wn in (named or {}):
                    inputs[inp["name"]] = named[wn]
            # Primitive nodes without an inputs list: use named values directly
            if not (n.get("inputs")):
                for k, v in (named or {}).items():
                    inputs.setdefault(k, v)
        else:
            wv = n.get("widgets_values") or []
            wi = 0
            for inp in (n.get("inputs") or []):
                w = inp.get("widget")
                if w and inp["name"] not in inputs:
                    if wi < len(wv):
                        inputs[inp["name"]] = wv[wi]
                        wi += 1
        # nodes with no inputs section but widgets (e.g. SaveVideo? has inputs)
        # Also handle nodes whose widgets are not in inputs (Primitive nodes):
        # their "value" widget becomes inputs["value"] if not linked
        if "widgets_values_named" in n and not any(
                i.get("widget") for i in (n.get("inputs") or [])):
            for k, v in (n.get("widgets_values_named") or {}).items():
                inputs[k if k != "value" else "value"] = v
        if class_type == "MiniMaxH3ReferenceToVideo":
            if "ref_image_size" not in inputs and named and "ref_image_size" in named:
                inputs["ref_image_size"] = named["ref_image_size"]
        if class_type == "BasicScheduler":
            inputs.setdefault("scheduler", "simple")
            inputs.setdefault("denoise", 1)
        prompt[nid] = {"class_type": class_type, "inputs": inputs}
    # SaveVideo filename_prefix null handling already in widgets_named? check
    sv = prompt.get("92")
    if sv and "filename_prefix" not in sv["inputs"]:
        sv["inputs"]["filename_prefix"] = None
    if sv:
        sv["inputs"].setdefault("format", "auto")
        sv["inputs"].setdefault("codec", "auto")
    return prompt


def validate(prompt):
    check = copy.deepcopy(prompt)
    check["138"]["inputs"]["value"] = "validation probe"
    check["132"]["inputs"]["value"] = 5.0
    check["129"]["inputs"]["noise_seed"] = 1
    check["92"]["inputs"]["filename_prefix"] = "probe/r2v_validate"
    data = json.dumps({"prompt": check}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit("VALIDATION FAILED:\n" + body[:8000])
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
            print("cancelled", pid)
        except Exception as e:
            print("cancel note", e)


def main():
    prompt = build()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(prompt, f, indent=2)
    print("Wrote", DST, len(prompt), "nodes")
    if "--validate" in sys.argv:
        validate(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
