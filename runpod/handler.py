"""
Runpod Serverless worker — multi-model image-to-video.

Input JSON (event["input"]):
  - image_base64 (required)
  - prompt (required)
  - model (optional): profile name from config/models.yaml
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import runpod

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "animegen").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from animegen.config import load_models_config  # noqa: E402
from animegen.engines import create_engine  # noqa: E402
from animegen.image_utils import image_from_base64, video_to_base64  # noqa: E402

_engines: dict[str, object] = {}


def _default_profile() -> str:
    return load_models_config().get("default_profile", "hunyuan15_i2v")


def get_engine(profile: str):
    if profile not in _engines:
        _engines[profile] = create_engine(profile)
    return _engines[profile]


def handler(event: dict) -> dict:
    job_input = event.get("input") or {}
    image_b64 = job_input.get("image_base64")
    prompt = job_input.get("prompt")
    if not image_b64 or not prompt:
        return {"error": "image_base64 and prompt are required"}

    profile = job_input.get("model") or os.environ.get("ANIMEGEN_MODEL") or _default_profile()

    try:
        image = image_from_base64(image_b64)
        engine = get_engine(profile)
        result = engine.generate(
            image,
            prompt,
            seed=job_input.get("seed"),
            num_frames=job_input.get("num_frames"),
            num_inference_steps=job_input.get("num_inference_steps"),
            width=job_input.get("width"),
            height=job_input.get("height"),
            guidance_scale=job_input.get("guidance_scale"),
            fps=job_input.get("fps"),
        )
        return {
            "video_base64": video_to_base64(result.video_path),
            "width": result.width,
            "height": result.height,
            "num_frames": result.num_frames,
            "fps": result.fps,
            "profile": result.profile,
            "engine": result.engine,
            "has_audio": result.has_audio,
        }
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
