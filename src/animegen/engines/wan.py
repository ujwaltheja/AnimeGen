from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

from animegen.config import ModelProfile, default_negative_prompt
from animegen.engines._load import place_pipeline
from animegen.engines.base import GenerateResult
from animegen.wan_image import fit_image_for_wan


class WanEngine:
    def __init__(self, profile_cfg: ModelProfile, device: str | None = None) -> None:
        self.profile_cfg = profile_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: WanImageToVideoPipeline | None = None

    def _load(self) -> WanImageToVideoPipeline:
        if self._pipe is not None:
            return self._pipe

        cfg = self.profile_cfg
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        if cfg.separate_vae:
            vae = AutoencoderKLWan.from_pretrained(
                cfg.model_id, subfolder="vae", torch_dtype=torch.float32
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                cfg.model_id, vae=vae, torch_dtype=dtype
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)

        if cfg.vae_tiling:
            pipe.vae.enable_tiling()

        place_pipeline(pipe, cfg, self.device)
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
        fps: float | None = None,
        seed: int | None = None,
        output_path: Path | None = None,
    ) -> GenerateResult:
        pipe = self._load()
        cfg = self.profile_cfg

        height = height if height is not None else cfg.default_height
        width = width if width is not None else cfg.default_width
        image, height, width = fit_image_for_wan(
            image, pipe, max_area=cfg.max_area, height=height, width=width
        )

        num_frames = num_frames or cfg.default_num_frames
        num_inference_steps = num_inference_steps or cfg.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.default_guidance
        fps_val = float(fps if fps is not None else cfg.default_fps)
        negative_prompt = negative_prompt or default_negative_prompt()

        gen_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = None
        if seed is not None:
            generator = torch.Generator(device=gen_device).manual_seed(seed)

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
        output_path = _temp_mp4(output_path)
        export_to_video(output, str(output_path), fps=int(fps_val))
        return GenerateResult(
            video_path=output_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps_val,
            profile=cfg.name,
            engine=cfg.engine,
        )


def _temp_mp4(output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = Path(tmp.name)
    tmp.close()
    return path
