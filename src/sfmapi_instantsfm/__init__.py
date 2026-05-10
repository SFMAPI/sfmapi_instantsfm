from __future__ import annotations

from .backend import InstantSfMBackend

try:
    from app.adapters.registry import register_backend
except ModuleNotFoundError:  # pragma: no cover
    register_backend = None  # type: ignore[assignment]

if register_backend is not None:
    register_backend("instantsfm", lambda: InstantSfMBackend())

__all__ = ["InstantSfMBackend"]
