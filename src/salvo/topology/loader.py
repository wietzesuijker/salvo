"""YAML loading with `extends:` inheritance, validated against schema."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from salvo.errors import ClusterYAMLError
from salvo.topology.schema import Cluster

_PRESETS_PKG = "salvo.topology.presets"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ClusterYAMLError(f"{path}: {e}") from e


def _find(name: str, search_dirs: list[Path]) -> Path:
    for d in search_dirs:
        candidate = d / f"{name}.yaml"
        if candidate.exists():
            return candidate
    raise ClusterYAMLError(f"cluster '{name}' not found in {search_dirs}")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Override wins for scalars; lists concatenate (override appended)."""
    out = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = list(out[k]) + list(v)
        else:
            out[k] = v
    return out


def _resolve(name: str, search_dirs: list[Path], seen: set[str]) -> dict[str, Any]:
    if name in seen:
        raise ClusterYAMLError(f"circular extends chain via {name}")
    seen.add(name)
    raw = _read_yaml(_find(name, search_dirs))
    parent_name = raw.get("extends")
    if parent_name:
        parent = _resolve(parent_name, search_dirs, seen)
        return _merge(parent, raw)
    return raw


def load_from_path(path: Path, search_dirs: list[Path] | None = None) -> Cluster:
    search_dirs = list(search_dirs or [path.parent])
    raw = _read_yaml(path)
    if "extends" in raw:
        parent = _resolve(raw["extends"], search_dirs, set())
        raw = _merge(parent, raw)
    raw.pop("extends", None)
    try:
        return Cluster.model_validate(raw)
    except ValidationError as e:
        raise ClusterYAMLError(f"{path}: {e}") from e


def load_cluster(cluster_id: str) -> Cluster:
    """Load a bundled preset by ID."""
    presets_dir = Path(str(files(_PRESETS_PKG)))
    return load_from_path(presets_dir / f"{cluster_id}.yaml", search_dirs=[presets_dir])
