from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from animegen.config import ModelProfile, get_profile


@dataclass
class GenerateResult:
    video_path: Path
    width: int
    height: int
    num_frames: int
    fps: float
    profile: str
    engine: str
    has_audio: bool = False


class I2VEngine(Protocol):
    profile_cfg: ModelProfile

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        fps: float | None = None,
        seed: int | None = None,
        output_path: Path | None = None,
    ) -> GenerateResult: ...


def create_engine(profile: str | None = None, device: str | None = None) -> I2VEngine:
    cfg = get_profile(profile)
    if cfg.engine == "wan":
        from animegen.engines.wan import WanEngine

        return WanEngine(cfg, device=device)
    if cfg.engine == "hunyuan15":
        from animegen.engines.hunyuan15 import Hunyuan15Engine

        return Hunyuan15Engine(cfg, device=device)
    if cfg.engine == "hunyuan":
        from animegen.engines.hunyuan import HunyuanEngine

        return HunyuanEngine(cfg, device=device)
    if cfg.engine == "ltx2":
        from animegen.engines.ltx2 import LTX2Engine

        return LTX2Engine(cfg, device=device)
    raise ValueError(f"Unknown engine '{cfg.engine}' for profile {cfg.name}")
