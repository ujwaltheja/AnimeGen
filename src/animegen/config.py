from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = ROOT / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    engine: str
    model_id: str
    license: str
    min_vram_gb: int
    recommended_vram_gb: int
    vae_tiling: bool
    prefer_cpu_offload: bool
    separate_vae: bool
    max_area: int | None
    default_height: int | None
    default_width: int | None
    default_fps: float
    default_num_frames: int
    default_steps: int
    default_guidance: float | None
    distilled: bool
    features: tuple[str, ...] = field(default_factory=tuple)
    pipeline: str | None = None


def load_models_config() -> dict:
    with MODELS_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_profile_names() -> list[str]:
    cfg = load_models_config()
    return sorted(cfg["profiles"].keys())


def get_profile(name: str | None = None) -> ModelProfile:
    cfg = load_models_config()
    default_name = cfg.get("default_profile", "hunyuan15_i2v")
    profile_name = name or os.environ.get("ANIMEGEN_MODEL", default_name)
    profiles = cfg["profiles"]
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile_name}'. Available: {available}")

    raw: dict[str, Any] = dict(profiles[profile_name])
    raw.setdefault("recommended_vram_gb", raw.get("min_vram_gb", 24))
    raw.setdefault("separate_vae", False)
    raw.setdefault("max_area", None)
    raw.setdefault("prefer_cpu_offload", False)
    raw.setdefault("vae_tiling", False)
    raw.setdefault("distilled", False)
    raw.setdefault("pipeline", None)
    feats = raw.pop("features", None) or []
    raw["features"] = tuple(feats)
    return ModelProfile(name=profile_name, **raw)


def default_negative_prompt() -> str:
    cfg = load_models_config()
    return cfg["default_negative_prompt"].replace("\n", "").strip()
