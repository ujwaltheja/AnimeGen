from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

import requests
from PIL import Image

ImageInput = Union[str, Path, Image.Image, bytes]


def load_image(source: ImageInput) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source)).convert("RGB")
    path = Path(source)
    if path.exists():
        return Image.open(path).convert("RGB")
    url = str(source)
    if url.startswith(("http://", "https://")):
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise FileNotFoundError(f"Image not found: {source}")


def image_from_base64(data: str) -> Image.Image:
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return load_image(raw)


def video_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
