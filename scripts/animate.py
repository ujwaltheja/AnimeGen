#!/usr/bin/env python3
"""Run image-to-video locally on a CUDA GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animegen.config import list_profile_names  # noqa: E402
from animegen.engines import create_engine  # noqa: E402
from animegen.image_utils import load_image  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Animate an image (multi-model I2V)")
    parser.add_argument("image", help="Path or URL to input image")
    parser.add_argument("prompt", help="Motion / scene description")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/animation.mp4"))
    parser.add_argument(
        "--model",
        choices=list_profile_names(),
        default=None,
        help="Profile name (default: ANIMEGEN_MODEL / config default_profile)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image)
    engine = create_engine(args.model)
    result = engine.generate(
        image,
        args.prompt,
        output_path=args.output,
        seed=args.seed,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
    )
    audio_note = " + audio" if result.has_audio else ""
    print(
        f"Saved {result.video_path} ({result.width}x{result.height}, "
        f"{result.num_frames} frames @ {result.fps} fps){audio_note}"
    )
    print(f"Profile: {result.profile} (engine={result.engine})")


if __name__ == "__main__":
    main()
