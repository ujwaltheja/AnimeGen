"""
Runpod Serverless worker for AnimeGen (Wan 2.2 image-to-video).

Input JSON (event["input"]):
  - image_base64 (required): PNG/JPEG as base64
  - prompt (required): text describing motion
  - model (optional): ti2v_5b | i2v_a14b
  - seed, num_frames, num_inference_steps, width, height (optional)
"""

from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path

import runpod

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "animegen").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from animegen.image_utils import image_from_base64, video_to_base64  # noqa: E402
from animegen.inference import WanI2VEngine  # noqa: E402

_engines: dict[str, WanI2VEngine] = {}


def get_engine(profile: str) -> WanI2VEngine:
    if profile not in _engines:
        _engines[profile] = WanI2VEngine(profile=profile)
    return _engines[profile]


def handler(event: dict) -> dict:
    job_input = event.get("input") or {}
    image_b64 = job_input.get("image_base64")
    prompt = job_input.get("prompt")
    if not image_b64 or not prompt:
        return {"error": "image_base64 and prompt are required"}

    profile = job_input.get("model") or os.environ.get("ANIMEGEN_MODEL", "ti2v_5b")

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
        }
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
