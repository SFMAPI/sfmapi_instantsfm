from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from sfmapi_instantsfm.backend import InstantSfMBackend
from sfmapi_instantsfm.plugin import get_plugin_manifest, plugin


def test_plugin_manifest_matches_hub_contract() -> None:
    manifest = get_plugin_manifest()

    assert manifest["plugin_id"] == "instantsfm"
    assert manifest["entry_points"] == ["sfmapi_instantsfm.plugin:plugin"]
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["instantsfm"]


def test_pyproject_declares_sfmapi_backend_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    entry_points = pyproject["project"]["entry-points"]["sfmapi.backends"]
    assert entry_points["instantsfm"] == "sfmapi_instantsfm.plugin:plugin"


def test_configured_entry_point_imports_plugin_object() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["entry-points"]["sfmapi.backends"]["instantsfm"]
    module_name, object_name = target.split(":", maxsplit=1)

    loaded = getattr(import_module(module_name), object_name)

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "instantsfm"


def test_plugin_registers_backend_factory() -> None:
    registered: dict[str, object] = {}

    plugin.register(lambda name, factory: registered.update({name: factory}))

    factory = registered["instantsfm"]
    assert callable(factory)
    assert isinstance(factory(), InstantSfMBackend)
