#!/usr/bin/env python3
"""Runpod Serverless worker (repo root — required for Runpod Hub builds)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from animegen.hub_handler import run_serverless  # noqa: E402

if __name__ == "__main__":
    run_serverless()
