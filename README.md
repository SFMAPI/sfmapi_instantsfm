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

The framing matters, so it is stated precisely:

- **The wrapper + SDK material in this package is plain
  `AGPL-3.0-or-later`, with no additional restrictions** (see
  `LICENSE`). This package adds **no** non-commercial term — doing so
  would be incoherent, since AGPLv3 §7 does not permit adding a
  field-of-use restriction (the same reason CC-BY-NC and the GPL
  family are incompatible). Use the wrapper under AGPL freely. The
  published package ships only this wrapper (`src/sfmapi_instantsfm/`)
  and references the upstream as a git submodule — it does **not**
  redistribute the upstream source.
- **Upstream InstantSfM** (`cre185/InstantSfM`, the
  `third_party/instantsfm` submodule) is **`CC-BY-NC-4.0` —
  non-commercial**. That limitation is **upstream's, not this
  project's**, and it binds whoever *operates* InstantSfM. Wrapping it
  neither adds nor removes that obligation. See
  `LICENSES/InstantSfM-CC-BY-NC-4.0.txt` and `THIRD_PARTY_NOTICES.md`.
- **sfmapi simply does not extend its commercial / dual license to
  this plugin** (see `LICENSING.md` in the sfmapi server repo) — a
  statement about the scope of sfmapi's *offer*, not a prohibition
  imposed here. A commercial sfmapi license cannot usefully cover a
  plugin whose upstream is non-commercial, so it doesn't.

Net: the wrapper is unrestricted AGPL; whether you may run InstantSfM
through it for commercial advantage is governed entirely by upstream
`CC-BY-NC-4.0`, on you as the operator. This plugin exists as a
demonstration of the action-catalog integration.
