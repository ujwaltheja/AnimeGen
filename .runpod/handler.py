#!/usr/bin/env python3
"""
Runpod Serverless handler (Hub entrypoint).

The Hub scanner expects handler.py inside .runpod/. This module re-exports
the shared handler used by the repo-root handler.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animegen.hub_handler import handler, run_serverless  # noqa: F401

if __name__ == "__main__":
    run_serverless()
