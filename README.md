# sfmapi InstantSfM backend

This package runs sfmapi with an InstantSfM-backed action catalog. The wrapper is AGPL-3.0-or-later; the upstream InstantSfM project is included as a git submodule under `third_party/instantsfm` and keeps its own CC-BY-NC-4.0 license.

The integration is intentionally action-based. InstantSfM has its own Python pipeline and does not expose sfmapi's portable feature/match/map storage contracts, so clients should discover and run native actions from `/v1/backend/actions`.

## Layout

- `src/sfmapi_instantsfm/`: Python backend adapter and sfmapi launcher.
- `third_party/instantsfm/`: upstream InstantSfM submodule.
- `tests/`: lightweight contract and HTTP discovery tests.
- `LICENSES/`: copied upstream license notice.

## Setup

```powershell
git submodule update --init --recursive
uv venv
uv sync --extra dev --extra mcp --with-editable ..\sfmapi
```

Install InstantSfM's heavy runtime dependencies in the environment you use for real jobs. The upstream project currently expects CUDA/NVIDIA GPU support and recommends Linux.

## Run sfmapi

```powershell
uv run sfmapi-instantsfm-api --instantsfm-root .\third_party\instantsfm --mcp local
```

Useful environment variables:

- `SFMAPI_INSTANTSFM_ROOT`: path to the upstream InstantSfM checkout.
- `SFMAPI_INSTANTSFM_PYTHON`: Python executable used to run InstantSfM modules.
- `SFMAPI_MCP_MODE=local`: mount sfmapi's MCP endpoint at `/mcp`.

The launcher configures an in-memory sfmapi demo server: SQLite memory DB, memory blob storage, inline queue, and inline tasks.

## Native Actions

Discover actions:

```powershell
curl "http://127.0.0.1:8000/v1/backend/actions?include_schemas=true"
```

Primary actions:

- `instantsfm.extractFeatures`: runs `ins-feat` semantics through `python -m instantsfm.scripts.feat`.
- `instantsfm.runGlobalSfm`: runs global SfM through `python -m instantsfm.scripts.sfm`.
- `instantsfm.trainGaussianSplatting`: runs `python -m instantsfm.scripts.gs`.
- `instantsfm.visualizeReconstruction`: runs `python -m instantsfm.scripts.vis_recon`.
- `instantsfm.runPipeline`: runs feature extraction, SfM, and optionally 3DGS in order.

Example action input:

```json
{
  "data_path": "C:/data/project",
  "feature_handler": "colmap",
  "manual_config_name": "colmap",
  "export_txt": true
}
```

## Tests

```powershell
uv run pytest -q
uv run ruff check src tests
```

The default tests mock subprocess execution and do not require CUDA, COLMAP, or InstantSfM dependencies.

## License

The sfmapi wrapper code is licensed under `AGPL-3.0-or-later`; see `LICENSE`.
Upstream InstantSfM is included as a submodule under `third_party/instantsfm`
and remains `CC-BY-NC-4.0`; see `LICENSES/InstantSfM-CC-BY-NC-4.0.txt` and
`THIRD_PARTY_NOTICES.md`. The upstream license is noncommercial, so confirm
terms before commercial use or redistribution.
