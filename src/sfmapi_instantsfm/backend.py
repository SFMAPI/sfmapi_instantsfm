from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from app.core.errors import CapabilityUnavailableError, NotFoundError, ValidationError
except ModuleNotFoundError:  # pragma: no cover - allows adapter tests without sfmapi installed

    class CapabilityUnavailableError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, *, capability: str, reason: str = "") -> None:
            super().__init__(reason or capability)

    class NotFoundError(RuntimeError):  # type: ignore[no-redef]
        pass

    class ValidationError(RuntimeError):  # type: ignore[no-redef]
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANTSFM_ROOT = REPO_ROOT / "third_party" / "instantsfm"
FEATURE_HANDLERS = (
    "colmap",
    "dedode",
    "disk+lightglue",
    "superpoint+lightglue",
    "sift",
)


@dataclass(frozen=True)
class InstantSfMCommand:
    action_id: str
    display_name: str
    category: str
    module: str
    description: str
    gpu_required: bool = True


INSTANTSFM_COMMANDS: tuple[InstantSfMCommand, ...] = (
    InstantSfMCommand(
        "instantsfm.extractFeatures",
        "InstantSfM feature extraction",
        "features",
        "instantsfm.scripts.feat",
        "Extract and match features into the InstantSfM/COLMAP database.",
    ),
    InstantSfMCommand(
        "instantsfm.runGlobalSfm",
        "InstantSfM global SfM",
        "mapping",
        "instantsfm.scripts.sfm",
        "Run InstantSfM global mapping and write a sparse reconstruction.",
    ),
    InstantSfMCommand(
        "instantsfm.trainGaussianSplatting",
        "InstantSfM 3DGS training",
        "dense",
        "instantsfm.scripts.gs",
        "Train the optional Gaussian Splatting viewer output.",
    ),
    InstantSfMCommand(
        "instantsfm.visualizeReconstruction",
        "InstantSfM reconstruction visualizer",
        "visualization",
        "instantsfm.scripts.vis_recon",
        "Open the InstantSfM offline reconstruction visualizer.",
        gpu_required=False,
    ),
)
_COMMAND_BY_ACTION = {command.action_id: command for command in INSTANTSFM_COMMANDS}
_SCRIPT_MODULES = {command.module for command in INSTANTSFM_COMMANDS}


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()


def resolve_instantsfm_root(value: str | Path | None) -> Path | None:
    raw = value or os.environ.get("SFMAPI_INSTANTSFM_ROOT")
    path = _expand_path(raw) if raw else DEFAULT_INSTANTSFM_ROOT
    if (path / "pyproject.toml").exists() and (path / "instantsfm").is_dir():
        return path.resolve()
    return None


def configure_instantsfm_environment(
    root: str | Path | None = None,
    *,
    python_executable: str | Path | None = None,
    validate: bool = False,
) -> Path | None:
    resolved_root = resolve_instantsfm_root(root)
    if resolved_root is None:
        if validate:
            raise ValueError(
                "InstantSfM checkout not found. Set SFMAPI_INSTANTSFM_ROOT or pass "
                "--instantsfm-root to sfmapi-instantsfm-api."
            )
        return None

    os.environ["SFMAPI_INSTANTSFM_ROOT"] = str(resolved_root)
    python = Path(python_executable or os.environ.get("SFMAPI_INSTANTSFM_PYTHON") or sys.executable)
    os.environ["SFMAPI_INSTANTSFM_PYTHON"] = str(python)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(resolved_root) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(resolved_root), *parts])
    return resolved_root


class InstantSfMBackend:
    name = "instantsfm"
    version = "0.1.0"
    vendor = "InstantSfM"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        python_executable: str | Path | None = None,
    ) -> None:
        self._root_override = _expand_path(root).resolve() if root else None
        self._python_executable = Path(
            python_executable or os.environ.get("SFMAPI_INSTANTSFM_PYTHON") or sys.executable
        )

    def capabilities(self) -> set[str]:
        return set()

    def runtime_versions(self) -> dict[str, str]:
        root = self._find_root()
        versions = {
            "backend": self.version,
            "instantsfm_root": str(root) if root else "missing",
            "instantsfm_python": str(self._python_executable),
        }
        if root is not None:
            commit = self._git_revision(root)
            if commit:
                versions["instantsfm_commit"] = commit
        return versions

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        actions = [self._pipeline_action(include_schemas=include_schemas)]
        actions.extend(
            self._command_action(command, include_schemas=include_schemas)
            for command in INSTANTSFM_COMMANDS
        )
        actions.append(self._module_action(include_schemas=include_schemas))
        return sorted(actions, key=lambda action: str(action["action_id"]))

    def get_backend_action(self, action_id: str) -> dict[str, Any]:
        for action in self.list_backend_actions(include_schemas=True):
            if action["action_id"] == action_id:
                return action
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def validate_backend_action(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        except ValidationError as exc:
            return {
                "action_id": action_id,
                "valid": False,
                "errors": [{"field": None, "message": str(exc)}],
                "normalized_inputs": {},
            }
        return {
            "action_id": action_id,
            "valid": True,
            "errors": [],
            "normalized_inputs": normalized,
        }

    def run_backend_action(
        self,
        action_id: str,
        inputs: dict[str, Any],
        *,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        if action_id == "instantsfm.runPipeline":
            return self._run_pipeline(normalized, workspace=workspace, progress=progress)
        if action_id == "instantsfm.runModule":
            return self._run_module_action(normalized)
        command = _COMMAND_BY_ACTION.get(action_id)
        if command is None:
            raise NotFoundError(f"Backend action {action_id!r} not found")
        return self._run_command(command, normalized, progress=progress)

    def extract_features(self, **_: Any) -> dict:
        raise self._unsupported("features.extract", "Use instantsfm.extractFeatures")

    def match(self, **_: Any) -> dict:
        raise self._unsupported("matches.exhaustive", "Use instantsfm.extractFeatures")

    def verify_matches(self, **_: Any) -> dict:
        raise self._unsupported("matches.verify", "Use instantsfm.extractFeatures")

    def read_keypoints(self, **_: Any) -> tuple[list[list[float]], bytes, int]:
        raise self._unsupported("observations.by_image", "InstantSfM does not expose keypoints")

    def iter_two_view_geometries(self, **_: Any) -> Iterable[tuple[int, int, Any]]:
        raise self._unsupported("matches.verify", "InstantSfM does not expose pair geometry")

    def iter_correspondences(self, **_: Any) -> Iterable[tuple[int, int, Any]]:
        raise self._unsupported("matches.exhaustive", "InstantSfM does not expose raw matches")

    def run_mapping(self, **_: Any) -> tuple[list[dict], list[Any]]:
        raise self._unsupported("map.global", "Use instantsfm.runGlobalSfm")

    def bundle_adjustment(self, **_: Any) -> dict:
        raise self._unsupported("ba.standard", "InstantSfM owns BA internally")

    def triangulate(self, **_: Any) -> dict:
        raise self._unsupported("triangulate.retri", "InstantSfM owns triangulation internally")

    def relocalize(self, **_: Any) -> dict:
        raise self._unsupported("relocalize.images")

    def pose_graph_optimize(self, **_: Any) -> dict:
        raise self._unsupported("pgo.optimize")

    def export(self, **_: Any) -> dict:
        raise self._unsupported("export.colmap_text", "Use instantsfm.runGlobalSfm with export_txt")

    def convert_spherical_to_cubemap(self, **_: Any) -> dict:
        raise self._unsupported("spherical.to_cubemap")

    def render_spherical_cubemap_images(self, **_: Any) -> dict:
        raise self._unsupported("spherical.render_cubemap")

    def build_vlad_index(self, **_: Any) -> tuple[list[str], Any]:
        raise self._unsupported("similarity.vlad")

    def localize_from_memory(self, **_: Any) -> dict:
        raise self._unsupported("localize.from_memory")

    def apply_sim3(self, **_: Any) -> dict:
        raise self._unsupported("georegister.sim3")

    def read_reconstruction(self, path: Path) -> Any:
        raise self._unsupported("import.colmap", f"Use sfmapi import tools for model path: {path}")

    def _find_root(self) -> Path | None:
        if self._root_override is not None:
            if (self._root_override / "pyproject.toml").exists():
                return self._root_override
            return None
        return resolve_instantsfm_root(None)

    def _require_root(self) -> Path:
        root = self._find_root()
        if root is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "InstantSfM checkout not found. Run `git submodule update --init "
                    "--recursive` and set SFMAPI_INSTANTSFM_ROOT if needed."
                ),
            )
        return root

    def _git_revision(self, root: Path) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except OSError:
            return None
        value = completed.stdout.strip()
        return value or None

    def _run_pipeline(
        self,
        inputs: dict[str, Any],
        *,
        workspace: Path | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        if workspace is not None:
            workspace.mkdir(parents=True, exist_ok=True)
        steps: list[tuple[InstantSfMCommand, dict[str, Any]]] = [
            (_COMMAND_BY_ACTION["instantsfm.extractFeatures"], inputs),
            (_COMMAND_BY_ACTION["instantsfm.runGlobalSfm"], inputs),
        ]
        if inputs.get("run_gaussian_splatting"):
            steps.append((_COMMAND_BY_ACTION["instantsfm.trainGaussianSplatting"], inputs))

        results: list[dict[str, Any]] = []
        total = len(steps)
        for index, (command, command_inputs) in enumerate(steps, start=1):
            self._progress(progress, command.category, index - 1, total)
            results.append(self._run_command(command, command_inputs, progress=None))
            self._progress(progress, command.category, index, total)
        return {
            "steps": results,
            "data_path": str(inputs["data_path"]),
            "sparse_path": str(Path(str(inputs["data_path"])) / "sparse"),
        }

    def _run_command(
        self,
        command: InstantSfMCommand,
        inputs: dict[str, Any],
        *,
        progress: Any | None,
    ) -> dict[str, Any]:
        module_args = self._module_args(command.action_id, inputs)
        self._progress(progress, command.category, 0, 1)
        completed = self._run_python_module(
            command.module,
            module_args,
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        self._progress(progress, command.category, 1, 1)
        return {
            "action_id": command.action_id,
            "module": command.module,
            "args": [str(self._python_executable), "-m", command.module, *module_args],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _run_module_action(self, inputs: dict[str, Any]) -> dict[str, Any]:
        module = str(inputs["module"])
        args = [str(arg) for arg in inputs.get("args", [])]
        completed = self._run_python_module(
            module,
            args,
            cwd=Path(str(inputs["cwd"])) if inputs.get("cwd") else None,
            extra_env={str(k): str(v) for k, v in dict(inputs.get("env") or {}).items()},
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        return {
            "module": module,
            "args": [str(self._python_executable), "-m", module, *args],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _run_python_module(
        self,
        module: str,
        args: list[str],
        *,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_seconds: int | float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        root = self._require_root()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), *[part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]]
        )
        if extra_env:
            env.update(extra_env)
        try:
            return subprocess.run(
                [str(self._python_executable), "-m", module, *args],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(cwd or root),
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"InstantSfM command failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(f"InstantSfM command timed out after {timeout_seconds}s") from exc

    def _module_args(self, action_id: str, inputs: dict[str, Any]) -> list[str]:
        if action_id == "instantsfm.extractFeatures":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "manual_config_name")
            self._add_optional(args, inputs, "feature_handler")
            self._add_flag(args, inputs, "single_camera")
            self._add_flag(args, inputs, "camera_per_folder")
            return args
        if action_id == "instantsfm.runGlobalSfm":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "manual_config_name")
            self._add_optional(args, inputs, "record_path")
            for flag in (
                "enable_gui",
                "record_recon",
                "disable_depths",
                "disable_semantics",
                "export_txt",
            ):
                self._add_flag(args, inputs, flag)
            return args
        if action_id == "instantsfm.trainGaussianSplatting":
            return ["--data_path", str(inputs["data_path"])]
        if action_id == "instantsfm.visualizeReconstruction":
            args = ["--data_path", str(inputs["data_path"])]
            self._add_optional(args, inputs, "record")
            return args
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _pipeline_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "instantsfm.runPipeline",
            "backend": self.name,
            "display_name": "InstantSfM feature + global SfM pipeline",
            "description": "Run feature extraction, global SfM, and optional 3DGS training.",
            "category": "pipeline",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "instantsfm",
                "upstream_root": str(self._find_root() or DEFAULT_INSTANTSFM_ROOT),
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._pipeline_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _command_action(
        self,
        command: InstantSfMCommand,
        *,
        include_schemas: bool,
    ) -> dict[str, Any]:
        descriptor = {
            "action_id": command.action_id,
            "backend": self.name,
            "display_name": command.display_name,
            "description": command.description,
            "category": command.category,
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": command.gpu_required,
            "required_capabilities": [],
            "metadata": {
                "family": "instantsfm",
                "module": command.module,
                "upstream_root": str(self._find_root() or DEFAULT_INSTANTSFM_ROOT),
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._input_schema_for_action(command.action_id)
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _module_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "instantsfm.runModule",
            "backend": self.name,
            "display_name": "InstantSfM Python module",
            "description": "Run an allow-listed InstantSfM Python module with explicit args.",
            "category": "utility",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {"family": "instantsfm", "allowlist": sorted(_SCRIPT_MODULES)},
        }
        if include_schemas:
            descriptor["input_schema"] = self._module_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _input_schema_for_action(self, action_id: str) -> dict[str, Any]:
        base = self._common_input_schema()
        if action_id == "instantsfm.extractFeatures":
            base["properties"].update(
                {
                    "manual_config_name": {"type": "string"},
                    "feature_handler": {"type": "string", "enum": list(FEATURE_HANDLERS)},
                    "single_camera": {"type": "boolean", "default": False},
                    "camera_per_folder": {"type": "boolean", "default": False},
                }
            )
        elif action_id == "instantsfm.runGlobalSfm":
            base["properties"].update(
                {
                    "manual_config_name": {"type": "string"},
                    "enable_gui": {"type": "boolean", "default": False},
                    "record_recon": {"type": "boolean", "default": False},
                    "record_path": {"type": "string"},
                    "disable_depths": {"type": "boolean", "default": False},
                    "disable_semantics": {"type": "boolean", "default": False},
                    "export_txt": {"type": "boolean", "default": False},
                }
            )
        elif action_id == "instantsfm.visualizeReconstruction":
            base["properties"]["record"] = {"type": "string"}
        return base

    def _pipeline_input_schema(self) -> dict[str, Any]:
        schema = self._input_schema_for_action("instantsfm.extractFeatures")
        schema["properties"].update(
            self._input_schema_for_action("instantsfm.runGlobalSfm")["properties"]
        )
        schema["properties"]["run_gaussian_splatting"] = {"type": "boolean", "default": False}
        return schema

    def _common_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["data_path"],
            "properties": {
                "data_path": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
        }

    def _module_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["module"],
            "properties": {
                "module": {"type": "string", "enum": sorted(_SCRIPT_MODULES)},
                "args": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                },
                "timeout_seconds": {"type": "number"},
            },
        }

    def _run_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "returncode": {"type": "integer"},
                "args": {"type": "array", "items": {"type": "string"}},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
            },
        }

    def _normalize_action_inputs(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.get_backend_action(action_id)
        if action_id == "instantsfm.runModule":
            module = str(inputs.get("module") or "")
            if module not in _SCRIPT_MODULES:
                raise ValidationError(
                    f"module must be one of: {', '.join(sorted(_SCRIPT_MODULES))}"
                )
            args = inputs.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ValidationError("args must be an array of strings")
            inputs["args"] = [str(arg) for arg in args]
            return inputs

        if not inputs.get("data_path"):
            raise ValidationError("data_path is required")
        if "feature_handler" in inputs:
            feature_handler = str(inputs["feature_handler"])
            if feature_handler not in FEATURE_HANDLERS:
                raise ValidationError(
                    f"feature_handler must be one of: {', '.join(FEATURE_HANDLERS)}"
                )
            inputs["feature_handler"] = feature_handler
        return inputs

    def _add_optional(self, args: list[str], inputs: dict[str, Any], name: str) -> None:
        value = inputs.get(name)
        if value is not None and str(value) != "":
            args.extend([f"--{name}", str(value)])

    def _add_flag(self, args: list[str], inputs: dict[str, Any], name: str) -> None:
        if bool(inputs.get(name, False)):
            args.append(f"--{name}")

    def _progress(self, progress: Any | None, phase: str, current: int, total: int) -> None:
        if progress is None:
            return
        try:
            progress.phase_progress(f"instantsfm.{phase}", current=current, total=total)
        except Exception:
            return

    def _unsupported(self, capability: str, reason: str = "") -> CapabilityUnavailableError:
        return CapabilityUnavailableError(capability=capability, reason=reason)


__all__ = [
    "DEFAULT_INSTANTSFM_ROOT",
    "FEATURE_HANDLERS",
    "INSTANTSFM_COMMANDS",
    "InstantSfMBackend",
    "configure_instantsfm_environment",
    "resolve_instantsfm_root",
]
