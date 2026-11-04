#!/usr/bin/env python3
"""Legacy path; Hub and Docker use repo-root handler.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animegen.hub_handler import run_serverless  # noqa: E402

if __name__ == "__main__":
    run_serverless()
