"""Backward-compatible exports; prefer animegen.engines.create_engine."""

from animegen.engines.base import GenerateResult, create_engine


class _WanShim:
    """Deprecated: use create_engine('i2v_a14b') instead."""

    def __init__(self, profile: str | None = None, device: str | None = None):
        self._engine = create_engine(profile, device=device)

    def generate(self, *args, **kwargs):
        return self._engine.generate(*args, **kwargs)


def __getattr__(name: str):
    if name == "WanI2VEngine":
        return _WanShim
    raise AttributeError(name)


__all__ = ["GenerateResult", "create_engine", "WanI2VEngine"]
