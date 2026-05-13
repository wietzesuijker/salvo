# Contributing

## Adding a cluster

The fastest contribution is a new cluster preset. Each preset is a single YAML file under `src/salvo/topology/presets/`. The schema lives in `src/salvo/topology/schema.py` (pydantic v2, frozen, `extra="forbid"`).

1. Copy an existing preset closest to your cluster (DRAC: `rorqual.yaml`; campus: `mila.yaml`).
2. Edit accounts, partitions, GPU types, walltime caps, defaults.
3. Add an entry to `src/salvo/topology/detect.py` for the hostname pattern.
4. Open a PR with the YAML + a one-line entry in the README cluster table.

Tests will exercise your preset via the parametrized suite in `tests/unit/test_presets.py`.

## Dev setup

    git clone <repo>
    cd salvo
    uv sync --all-extras
    uv run pre-commit install
    uv run pytest

## Style

- Python 3.11+, ruff + mypy strict, line length 100.
- Pydantic v2 models are frozen and `extra="forbid"`.
- Public functions get type hints; tests get docstrings on intent (not behaviour).
- No mutable defaults, no f-strings in log messages.

## Test layers

- **Layer A** (`tests/unit/`): pure-Python, no subprocess.
- **Layer B** (`tests/integration/`): subprocess to fake `sbatch` via `tests/conftest.py:fake_sbatch`. Golden sbatch files in `tests/integration/golden/`.
- **Layer C** (cluster nightly): not run on PRs.
- **Perf** (`tests/perf/`): budget smoke tests.

PRs must keep total coverage at or above 80% and add tests for every new dispatch rule.

## Commit style

Conventional commits with module-scoped scope: `feat(job):`, `fix(dispatch):`, `test(integration):`, `docs:`, `chore:`.
