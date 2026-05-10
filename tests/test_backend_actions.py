from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sfmapi_instantsfm.backend import InstantSfMBackend


def _fake_instantsfm(root: Path) -> Path:
    (root / "instantsfm").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    return root


def test_action_catalog_exposes_instantsfm_actions(tmp_path: Path) -> None:
    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))

    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "instantsfm.extractFeatures" in action_ids
    assert "instantsfm.runGlobalSfm" in action_ids
    assert "instantsfm.runPipeline" in action_ids
    assert "instantsfm.runModule" in action_ids
    assert backend.capabilities() == set()
    extract = next(
        action for action in actions if action["action_id"] == "instantsfm.extractFeatures"
    )
    assert extract["input_schema"]["properties"]["feature_handler"]["enum"]


def test_backend_contract_passes(tmp_path: Path) -> None:
    pytest.importorskip("app.adapters.backend_contract")
    from app.adapters.backend_contract import assert_backend_contract

    assert_backend_contract(InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM")))


def test_validate_rejects_unknown_feature_handler(tmp_path: Path) -> None:
    backend = InstantSfMBackend(_fake_instantsfm(tmp_path / "InstantSfM"))

    result = backend.validate_backend_action(
        "instantsfm.extractFeatures",
        {"data_path": "dataset", "feature_handler": "unknown"},
    )

    assert result["valid"] is False
    assert "feature_handler must be one of" in result["errors"][0]["message"]


def test_run_extract_features_builds_module_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    backend = InstantSfMBackend(root, python_executable="python")
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "instantsfm.extractFeatures",
        {
            "data_path": "C:/data/project",
            "feature_handler": "colmap",
            "manual_config_name": "colmap",
            "single_camera": True,
        },
    )

    assert result["returncode"] == 0
    assert captured["args"] == [
        "python",
        "-m",
        "instantsfm.scripts.feat",
        "--data_path",
        "C:/data/project",
        "--manual_config_name",
        "colmap",
        "--feature_handler",
        "colmap",
        "--single_camera",
    ]


def test_run_pipeline_executes_ordered_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_instantsfm(tmp_path / "InstantSfM")
    backend = InstantSfMBackend(root, python_executable="python")
    modules: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        modules.append(args[2])
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "instantsfm.runPipeline",
        {"data_path": str(tmp_path / "dataset"), "export_txt": True},
    )

    assert modules == ["instantsfm.scripts.feat", "instantsfm.scripts.sfm"]
    assert len(result["steps"]) == 2
