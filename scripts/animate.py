#!/usr/bin/env python3
"""Run image-to-video locally on a CUDA GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animegen.image_utils import load_image  # noqa: E402
from animegen.inference import WanI2VEngine  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Animate an image with Wan 2.2 I2V")
    parser.add_argument("image", help="Path or URL to input image")
    parser.add_argument("prompt", help="Motion / scene description")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/animation.mp4"))
    parser.add_argument(
        "--model",
        choices=("ti2v_5b", "i2v_a14b"),
        default=None,
        help="Model profile (default: ANIMEGEN_MODEL or ti2v_5b)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image)
    engine = WanI2VEngine(profile=args.model)
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
    print(f"Saved {result.video_path} ({result.width}x{result.height}, {result.num_frames} frames @ {result.fps} fps)")
    print(f"Profile: {result.profile}")


if __name__ == "__main__":
    main()
