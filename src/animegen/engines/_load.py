from __future__ import annotations

import os

from animegen.config import ModelProfile


def resolve_load_mode(profile: ModelProfile) -> str:
    mode = os.environ.get("ANIMEGEN_LOAD_MODE", "").strip().lower()
    if mode in ("gpu", "offload"):
        return mode
    if profile.prefer_cpu_offload:
        return "offload"
    return "gpu"


def place_pipeline(pipe, profile: ModelProfile, device: str) -> None:
    load_mode = resolve_load_mode(profile)
    if load_mode == "offload":
        try:
            pipe.enable_model_cpu_offload(device=device if device.startswith("cuda") else "cuda")
        except TypeError:
            pipe.enable_model_cpu_offload()
    elif device.startswith("cuda"):
        pipe.to(device)
    else:
        pipe.enable_model_cpu_offload()
