"""ASGI entrypoint that registers the InstantSfM backend before sfmapi starts."""
# ruff: noqa: E402,I001

from __future__ import annotations

import os

os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "instantsfm")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sfmapi_instantsfm.backend import configure_instantsfm_environment

configure_instantsfm_environment(validate=bool(os.environ.get("SFMAPI_INSTANTSFM_ROOT")))

import sfmapi_instantsfm  # noqa: F401 - import side effect registers backend

from app.main import create_app

app = create_app()
