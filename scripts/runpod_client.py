#!/usr/bin/env python3
"""Call a Runpod Serverless endpoint for image-to-video."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Local image file")
    parser.add_argument("prompt", help="Motion prompt")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/runpod.mp4"))
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("ANIMEGEN_MODEL", "hunyuan15_i2v"))
    args = parser.parse_args()

    if not args.endpoint_id or not args.api_key:
        print("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in .env or pass flags.", file=sys.stderr)
        sys.exit(1)

    b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = {
        "input": {
            "image_base64": b64,
            "prompt": args.prompt,
            "model": args.model,
        }
    }
    url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=3600,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "COMPLETED":
        print(json.dumps(body, indent=2), file=sys.stderr)
        sys.exit(1)

    out = body["output"]
    video_b64 = out["video_base64"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(video_b64))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
