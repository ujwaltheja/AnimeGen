from __future__ import annotations

from pathlib import Path

import torch
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video
from PIL import Image

from animegen.config import ModelProfile
from animegen.engines._load import place_pipeline
from animegen.engines.base import GenerateResult
from animegen.engines.wan import _temp_mp4


class HunyuanEngine:
    """Original HunyuanVideo I2V (2025 checkpoint)."""

    def __init__(self, profile_cfg: ModelProfile, device: str | None = None) -> None:
        self.profile_cfg = profile_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: HunyuanVideoImageToVideoPipeline | None = None

    def _load(self) -> HunyuanVideoImageToVideoPipeline:
        if self._pipe is not None:
            return self._pipe

        cfg = self.profile_cfg
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            cfg.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            cfg.model_id, transformer=transformer, torch_dtype=torch.float16
        )
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

        num_frames = num_frames or cfg.default_num_frames
        num_inference_steps = num_inference_steps or cfg.default_steps
        fps_val = float(fps if fps is not None else cfg.default_fps)

        gen_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = None
        if seed is not None:
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        kwargs: dict = dict(
            image=image.convert("RGB"),
            prompt=prompt,
            num_inference_steps=num_inference_steps,
        )
        if num_frames:
            kwargs["num_frames"] = num_frames
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        elif cfg.default_guidance is not None:
            kwargs["guidance_scale"] = cfg.default_guidance
        if generator is not None:
            kwargs["generator"] = generator

        output = pipe(**kwargs).frames[0]
        out_path = _temp_mp4(output_path)
        export_to_video(output, str(out_path), fps=int(fps_val))

        return GenerateResult(
            video_path=out_path,
            width=width or image.width,
            height=height or image.height,
            num_frames=num_frames,
            fps=fps_val,
            profile=cfg.name,
            engine=cfg.engine,
        )
