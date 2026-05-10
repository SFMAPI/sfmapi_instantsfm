from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sfmapi_instantsfm.backend import InstantSfMBackend


def _fake_instantsfm(root: Path) -> Path:
    (root / "instantsfm").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    return root


def test_sfmapi_http_discovery_surfaces_instantsfm_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from app.adapters.registry import register_backend
    from app.core.capabilities import reset_capabilities_cache
    from app.core.config import reset_settings_for_tests
    from app.db.session import reset_engine_for_tests
    from fastapi.testclient import TestClient

    root = _fake_instantsfm(tmp_path / "InstantSfM")
    monkeypatch.setenv("SFMAPI_BACKEND", "instantsfm")
    monkeypatch.setenv("SFMAPI_MCP_MODE", "off")
    from app.main import create_app

    settings = reset_settings_for_tests(
        ephemeral=True,
        db_url="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
        blob_backend="memory",
        queue_backend="inline",
        inline_tasks=True,
        workspace_root=tmp_path / "workspace",
    )
    asyncio.run(reset_engine_for_tests(settings))
    reset_capabilities_cache()
    register_backend("instantsfm", lambda: InstantSfMBackend(root))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["backend"]["name"] == "instantsfm"
        assert capabilities["features"]["backend.actions"] is True
        assert capabilities["features"]["backend.config_schemas"] is False

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "instantsfm"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] == 0

        actions = client.get("/v1/backend/actions?include_schemas=true&page_size=50").json()[
            "items"
        ]
        action_ids = {action["action_id"] for action in actions}
        assert "instantsfm.runPipeline" in action_ids
        assert "instantsfm.extractFeatures" in action_ids
        pipeline = next(
            action for action in actions if action["action_id"] == "instantsfm.runPipeline"
        )
        assert "data_path" in pipeline["input_schema"]["properties"]

        assert client.get("/v1/backend/config-schemas").json()["items"] == []
