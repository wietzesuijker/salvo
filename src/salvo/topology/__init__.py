"""Public topology surface.

`ClusterTopology` is an alias for the internal `Cluster` pydantic model.
`load_preset` and `list_presets` wrap the YAML-backed preset loader.
"""

from __future__ import annotations

from importlib.resources import files

from salvo.topology.loader import load_cluster as load_preset
from salvo.topology.schema import (
    Account,
    Cluster,
    DispatchRule,
    LoginConstraints,
    Partition,
)

ClusterTopology = Cluster

_PRESETS_PKG = "salvo.topology.presets"


def list_presets() -> list[str]:
    """Return sorted preset IDs bundled with salvo (excludes private `_*` files)."""
    presets_dir = files(_PRESETS_PKG)
    return sorted(
        p.name[: -len(".yaml")]
        for p in presets_dir.iterdir()
        if p.name.endswith(".yaml") and not p.name.startswith("_")
    )


__all__ = [
    "Account",
    "Cluster",
    "ClusterTopology",
    "DispatchRule",
    "LoginConstraints",
    "Partition",
    "list_presets",
    "load_preset",
]
