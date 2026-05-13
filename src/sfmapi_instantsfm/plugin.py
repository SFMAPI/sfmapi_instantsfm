from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from .backend import InstantSfMBackend


class ProviderManifestDict(TypedDict):
    provider_id: str
    display_name: str
    capabilities: list[str]
    backend_actions: list[str]
    priority_hint: int


class PluginManifestDict(TypedDict):
    plugin_id: str
    display_name: str
    description: str
    package_name: str
    github_url: str
    entry_points: list[str]
    providers: list[ProviderManifestDict]
    runtime_modes: dict[str, Any]
    capabilities: list[str]
    backend_actions: list[str]
    config_schemas: list[str]
    artifact_contracts: list[str]
    licenses: list[dict[str, str]]
    upstream_projects: list[dict[str, str]]
    compatibility: dict[str, Any]
    conformance: dict[str, str]
    trust_tier: str


manifest: PluginManifestDict = {
    "plugin_id": "instantsfm",
    "display_name": "InstantSfM",
    "description": "Backend plugin for Python-based InstantSfM workflows.",
    "package_name": "sfmapi-instantsfm",
    "github_url": "https://github.com/SFMAPI/sfmapi_instantsfm.git",
    "entry_points": ["sfmapi_instantsfm.plugin:plugin"],
    "providers": [
        {
            "provider_id": "instantsfm",
            "display_name": "InstantSfM",
            "capabilities": [
                "features.extract.superpoint",
                "pairs.retrieval",
                "matchers.lightglue",
                "map.incremental",
            ],
            "backend_actions": ["instantsfm.*"],
            "priority_hint": 60,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SFMAPI/sfmapi_instantsfm.git",
            "ref": "main",
            "package": "sfmapi-instantsfm",
        },
        "docker": {},
    },
    "capabilities": [
        "features.extract.superpoint",
        "pairs.retrieval",
        "matchers.lightglue",
        "map.incremental",
    ],
    "backend_actions": ["instantsfm.*"],
    "config_schemas": ["instantsfm.*"],
    "artifact_contracts": [
        "sfmapi.features",
        "sfmapi.matches",
        "sfmapi.reconstruction",
    ],
    "licenses": [{"name": "AGPL-3.0-or-later"}],
    "upstream_projects": [
        {
            "name": "InstantSfM",
            "url": "https://github.com/cre185/InstantSfM",
            "license": "CC-BY-NC-4.0",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "recommended",
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "community",
}


def backend_factory() -> InstantSfMBackend:
    return InstantSfMBackend()


def get_plugin_manifest() -> PluginManifestDict:
    return manifest


def register(
    register_backend: Callable[[str, Callable[[], InstantSfMBackend]], None],
) -> None:
    register_backend("instantsfm", backend_factory)


@dataclass(frozen=True)
class SfmapiBackendPlugin:
    manifest: PluginManifestDict
    backend_name: str
    backend_factory: Callable[[], InstantSfMBackend]

    def get_plugin_manifest(self) -> PluginManifestDict:
        return self.manifest

    def register(
        self,
        register_backend: Callable[[str, Callable[[], InstantSfMBackend]], None],
    ) -> None:
        register_backend(self.backend_name, self.backend_factory)


plugin = SfmapiBackendPlugin(
    manifest=manifest,
    backend_name="instantsfm",
    backend_factory=backend_factory,
)


__all__ = [
    "PluginManifestDict",
    "SfmapiBackendPlugin",
    "backend_factory",
    "get_plugin_manifest",
    "manifest",
    "plugin",
    "register",
]
