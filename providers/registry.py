"""Extensible model registry.

Adding a model = add its workflow file + its provider class + one entry in
config/models.json. The pipeline (main.py) never changes.
"""
from __future__ import annotations

import importlib
import json
import os

from .errors import UnknownModelError

# provider id -> (module, class name). Imported lazily so registry metadata
# (lookup/listing) never requires model dependencies.
IMAGE_PROVIDERS = {
    "sd15": ("providers.image.sd15", "Sd15ImageProvider"),
    "sdxl": ("providers.image.sdxl", "SdxlImageProvider"),
    "flux": ("providers.image.flux", "FluxImageProvider"),
    "krea": ("providers.image.krea", "KreaImageProvider"),
}
VIDEO_PROVIDERS = {
    "ltx": ("providers.video.ltx", "LtxVideoProvider"),
    "wan": ("providers.video.wan", "WanVideoProvider"),
}
AUDIO_PROVIDERS = {
    "edge_tts": ("providers.audio.edge_tts", "EdgeTtsProvider"),
}

KIND_TABLES = {"image": IMAGE_PROVIDERS, "video": VIDEO_PROVIDERS, "audio": AUDIO_PROVIDERS}


def registry_path(base_dir: str) -> str:
    return os.path.join(base_dir, "config", "models.json")


def load_registry(base_dir: str) -> dict:
    path = registry_path(base_dir)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for kind in ("image", "video", "audio"):
        if not isinstance(data.get(kind), dict):
            raise ValueError(f"Registry {path} missing section: {kind!r}")
    return data


def lookup(base_dir: str, kind: str, model: str) -> dict:
    """Return the registry entry for a model. Raises UnknownModelError."""
    data = load_registry(base_dir)
    section = data.get(kind, {})
    if model not in section:
        available = ", ".join(sorted(section)) or "(none)"
        raise UnknownModelError(
            f"Unknown {kind} model {model!r}. Available: {available}.")
    entry = dict(section[model])
    entry["model"] = model
    wf = entry.get("workflow")
    if wf:
        entry["workflow_path"] = os.path.join(base_dir, wf)
    wfr = entry.get("workflow_ref")
    if wfr:
        entry["workflow_ref_path"] = os.path.join(base_dir, wfr)
    return entry


def list_models(base_dir: str, kind: str) -> list[str]:
    return sorted(load_registry(base_dir).get(kind, {}))


def _provider_class(kind: str, provider_id: str):
    table = KIND_TABLES[kind]
    if provider_id not in table:
        raise UnknownModelError(
            f"Unknown {kind} provider {provider_id!r}. "
            f"Available: {', '.join(sorted(table))}.")
    module_name, class_name = table[provider_id]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_provider(kind: str, base_dir: str, model: str, **kwargs):
    """Instantiate the provider registered for (kind, model)."""
    entry = lookup(base_dir, kind, model)
    cls = _provider_class(kind, entry["provider"])
    return cls(name=model, settings=entry, **kwargs)
