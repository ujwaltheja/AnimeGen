from __future__ import annotations

import math

from PIL import Image

from diffusers import WanImageToVideoPipeline


def fit_image_for_wan(
    image: Image.Image,
    pipe: WanImageToVideoPipeline,
    *,
    max_area: int | None,
    height: int | None,
    width: int | None,
) -> tuple[Image.Image, int, int]:
    """Resize input to Wan I2V-friendly dimensions (official HF recipe)."""
    if height is not None and width is not None:
        return image.resize((width, height), Image.Resampling.LANCZOS), height, width

    if height is not None or width is not None:
        raise ValueError("Provide both height and width, or neither (use max_area smart resize).")

    if max_area is None:
        raise ValueError("Profile has no max_area; pass explicit height and width.")

    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    aspect_ratio = image.height / image.width
    h = round(math.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    w = round(math.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    h = max(mod_value, h)
    w = max(mod_value, w)
    resized = image.resize((w, h), Image.Resampling.LANCZOS)
    return resized, h, w
