from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

from animegen.config import ModelProfile, default_negative_prompt, get_profile


@dataclass
class GenerateResult:
    video_path: Path
    width: int
    height: int
    num_frames: int
    fps: int
    profile: str


class WanI2VEngine:
    """Lazy-loaded Wan 2.2 image-to-video pipeline."""

    def __init__(self, profile: str | None = None, device: str | None = None) -> None:
        self.profile_cfg: ModelProfile = get_profile(profile)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: WanImageToVideoPipeline | None = None

    def _load(self) -> WanImageToVideoPipeline:
        if self._pipe is not None:
            return self._pipe

        model_id = self.profile_cfg.model_id
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=dtype
        )

        if self.device.startswith("cuda"):
            pipe.to(self.device)
        else:
            pipe.enable_model_cpu_offload()

        self._pipe = pipe
        return pipe

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
        fps: int | None = None,
        seed: int | None = None,
        output_path: Path | None = None,
    ) -> GenerateResult:
        pipe = self._load()
        cfg = self.profile_cfg

        height = height or cfg.default_height
        width = width or cfg.default_width
        num_frames = num_frames or cfg.default_num_frames
        num_inference_steps = num_inference_steps or cfg.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.default_guidance
        fps = fps or cfg.default_fps
        negative_prompt = negative_prompt or default_negative_prompt()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs: dict[str, Any] = dict(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        if generator is not None:
            kwargs["generator"] = generator

        output = pipe(**kwargs).frames[0]

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = Path(tmp.name)
            tmp.close()

        export_to_video(output, str(output_path), fps=fps)
        return GenerateResult(
            video_path=output_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            profile=cfg.name,
        )
