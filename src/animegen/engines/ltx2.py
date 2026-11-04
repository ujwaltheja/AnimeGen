from __future__ import annotations

from pathlib import Path

import torch
from diffusers import LTX2ImageToVideoPipeline
from PIL import Image

from animegen.config import ModelProfile
from animegen.engines._load import place_pipeline
from animegen.engines.base import GenerateResult
from animegen.engines.wan import _temp_mp4


def _ltx_encode_video(
    video,
    audio,
    *,
    fps: float,
    sample_rate: int,
    output_path: Path,
) -> None:
    try:
        from diffusers.pipelines.ltx2.export_utils import encode_video
    except ImportError:
        from diffusers.utils import encode_video  # type: ignore[attr-defined]

    encode_video(
        video[0],
        fps=fps,
        audio=audio[0].float().cpu(),
        audio_sample_rate=sample_rate,
        output_path=str(output_path),
    )


def _ltx_negative_prompt() -> str:
    try:
        from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

        return DEFAULT_NEGATIVE_PROMPT
    except ImportError:
        return "blurry, low quality, distorted, artifacts"


def _align_ltx_dims(width: int, height: int) -> tuple[int, int]:
    mod = 32
    return max(mod, width // mod * mod), max(mod, height // mod * mod)


def _align_ltx_frames(num_frames: int) -> int:
    if num_frames % 8 == 1:
        return num_frames
    return max(9, (num_frames // 8) * 8 + 1)


class LTX2Engine:
    """LTX-2.3 image-to-video with synchronized audio."""

    def __init__(self, profile_cfg: ModelProfile, device: str | None = None) -> None:
        self.profile_cfg = profile_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: LTX2ImageToVideoPipeline | None = None

    def _load(self) -> LTX2ImageToVideoPipeline:
        if self._pipe is not None:
            return self._pipe

        cfg = self.profile_cfg
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        pipe = LTX2ImageToVideoPipeline.from_pretrained(cfg.model_id, torch_dtype=dtype)
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

        width = width or cfg.default_width or 768
        height = height or cfg.default_height or 512
        width, height = _align_ltx_dims(width, height)
        num_frames = _align_ltx_frames(num_frames or cfg.default_num_frames)
        num_inference_steps = num_inference_steps or cfg.default_steps
        guidance_scale = guidance_scale if guidance_scale is not None else cfg.default_guidance
        frame_rate = float(fps if fps is not None else cfg.default_fps)
        negative_prompt = negative_prompt or _ltx_negative_prompt()

        gen_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = None
        if seed is not None:
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        pipe_kwargs: dict = dict(
            image=image.convert("RGB"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="np",
            return_dict=False,
        )
        if generator is not None:
            pipe_kwargs["generator"] = generator

        if cfg.distilled:
            try:
                from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
            except ImportError:
                DISTILLED_SIGMA_VALUES = None  # type: ignore[assignment]
            if DISTILLED_SIGMA_VALUES is not None:
                pipe_kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
                pipe_kwargs["guidance_scale"] = 1.0
                pipe_kwargs["num_inference_steps"] = 8
        else:
            pipe_kwargs.update(
                stg_scale=1.0,
                modality_scale=3.0,
                guidance_rescale=0.7,
                audio_guidance_scale=7.0,
                audio_stg_scale=1.0,
                audio_modality_scale=3.0,
                audio_guidance_rescale=0.7,
                spatio_temporal_guidance_blocks=[28],
            )

        video, audio = pipe(**pipe_kwargs)
        out_path = _temp_mp4(output_path)
        _ltx_encode_video(
            video,
            audio,
            fps=frame_rate,
            sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=out_path,
        )

        return GenerateResult(
            video_path=out_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=frame_rate,
            profile=cfg.name,
            engine=cfg.engine,
            has_audio=True,
        )
