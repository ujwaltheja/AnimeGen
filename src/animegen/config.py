from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_CONFIG = ROOT / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    pipeline: str
    license: str
    min_vram_gb: int
    default_height: int
    default_width: int
    default_fps: int
    default_num_frames: int
    default_steps: int
    default_guidance: float


def load_models_config() -> dict:
    with MODELS_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_profile(name: str | None = None) -> ModelProfile:
    cfg = load_models_config()
    profile_name = name or os.environ.get("ANIMEGEN_MODEL", "ti2v_5b")
    profiles = cfg["profiles"]
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile_name}'. Available: {available}")
    p = profiles[profile_name]
    return ModelProfile(name=profile_name, **p)


def default_negative_prompt() -> str:
    cfg = load_models_config()
    return cfg["default_negative_prompt"].replace("\n", "").strip()
