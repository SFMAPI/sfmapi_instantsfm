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
    "description": (
        "Wrapper + SDK material is plain AGPL-3.0-or-later with no "
        "added restrictions. Upstream InstantSfM (cre185/InstantSfM) "
        "is CC-BY-NC-4.0; that non-commercial term is upstream's and "
        "binds whoever runs InstantSfM, not added by sfmapi. sfmapi's "
        "commercial/dual license does not extend to this plugin."
    ),
    "package_name": "sfmapi-instantsfm",
    "github_url": "https://github.com/SFMAPI/sfmapi_instantsfm.git",
    "entry_points": ["sfmapi_instantsfm.plugin:plugin"],
    "providers": [
        {
            "provider_id": "instantsfm",
            "display_name": "InstantSfM",
            # InstantSfM is a *global* SfM engine. It backs one portable
            # capability -- `map.global` -- via a path-staging adapter:
            # `InstantSfMBackend.run_mapping` stages a temp project root
            # whose `database.db` / `images` entries link to sfmapi's
            # independent paths, runs `scripts.sfm` against it, and reads
            # the COLMAP sparse model back out. Feature extraction stays
            # action-only: `scripts.feat` fuses extraction + matching
            # into one whole-project `GenerateDatabase` call (no separate
            # extract/pairs/match stages, and it refuses to run if the
            # database already exists), which does not map to a thin
            # portable-stage wrapper. Everything else is exposed via
            # `instantsfm.*` backend actions.
            "capabilities": ["map.global"],
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
    # One portable capability -- `map.global` -- backed by the
    # `run_mapping` path-staging adapter. No portable config schemas and
    # no portable artifact contracts: those stay `[]` until backed by
    # real `list_backend_config_schemas` / `list_backend_artifact_
    # contracts` methods. The previously declared
    # `features.extract.superpoint`, `pairs.retrieval`, and
    # `matchers.lightglue` had no backing code and have been removed
    # rather than faked; `map.incremental` was also the wrong name --
    # InstantSfM is a *global* SfM engine, so the portable stage it now
    # backs is `map.global`.
    "capabilities": ["map.global"],
    "backend_actions": ["instantsfm.*"],
    "config_schemas": [],
    "artifact_contracts": [],
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


def register(register_backend: Callable[..., None]) -> None:
    provider_ids = [str(provider["provider_id"]) for provider in manifest["providers"]]
    try:
        register_backend("instantsfm", backend_factory, providers=provider_ids)
    except TypeError:
        # Older sfmapi without ``providers=`` kwarg on the registrar.
        register_backend("instantsfm", backend_factory)


@dataclass(frozen=True)
class SfmapiBackendPlugin:
    manifest: PluginManifestDict
    backend_name: str
    backend_factory: Callable[[], InstantSfMBackend]

    def get_plugin_manifest(self) -> PluginManifestDict:
        return self.manifest

    def register(self, register_backend: Callable[..., None]) -> None:
        provider_ids = [str(provider["provider_id"]) for provider in self.manifest["providers"]]
        try:
            register_backend(self.backend_name, self.backend_factory, providers=provider_ids)
        except TypeError:
            # Older sfmapi without ``providers=`` kwarg on the registrar.
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
