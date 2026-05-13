"""salvo doctor — self-check + fix hints.

Phase 0 ships these checks; more added per quirks codex.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from salvo.errors import ClusterYAMLError
from salvo.manifest.store import Manifest
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster


class CheckStatus(StrEnum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class Check:
    name: str
    status: CheckStatus
    message: str
    fix: str = ""


def _check_topology() -> list[Check]:
    cid = detect_cluster()
    if cid is None:
        return [
            Check(
                "topology.detect",
                CheckStatus.WARN,
                "no cluster detected",
                fix="export SALVO_CLUSTER=<id>",
            )
        ]
    try:
        cluster = load_cluster(cid)
    except ClusterYAMLError as e:
        return [
            Check(
                "topology.detect",
                CheckStatus.FAIL,
                f"preset for {cid!r} not valid: {e}",
            )
        ]
    return [
        Check(
            "topology.detect",
            CheckStatus.OK,
            f"detected cluster: {cid} ({cluster.display_name})",
        )
    ]


def _check_ssh(cid: str | None) -> list[Check]:
    if cid is None:
        return []
    try:
        cluster = load_cluster(cid)
    except ClusterYAMLError:
        return []
    if cluster.login is None:
        return [Check("ssh.alias", CheckStatus.WARN, f"no ssh_alias declared for {cid}")]
    alias = cluster.login.ssh_alias
    if "." in alias:
        return [
            Check(
                "ssh.alias",
                CheckStatus.FAIL,
                f"ssh_alias {alias!r} looks like an FQDN; use alias only to keep ControlPath valid",
                fix="set a non-FQDN ssh_alias in the cluster preset",
            )
        ]
    if not shutil.which("ssh"):
        return [Check("ssh.alias", CheckStatus.FAIL, "ssh not found in PATH")]
    return [Check("ssh.alias", CheckStatus.OK, f"ssh alias {alias!r} looks well-formed")]


def _check_manifest() -> list[Check]:
    p = Path(os.path.expanduser("~/.salvo/manifest.toml"))
    if not p.exists():
        return [
            Check(
                "manifest.exists",
                CheckStatus.WARN,
                f"no manifest at {p}; data gate will block all data_needs",
            )
        ]
    m = Manifest.load(p)
    stale = []
    cutoff = datetime.now(UTC) - timedelta(days=30)
    for name, ds in m.datasets.items():
        for cid, loc in ds.locations.items():
            if loc.verified_at < cutoff:
                stale.append(f"{name}@{cid}")
    if stale:
        return [
            Check(
                "manifest.fresh",
                CheckStatus.WARN,
                f"{len(stale)} stale manifest entries (>30 d): {', '.join(stale[:5])}",
                fix="re-run `salvo stage push` for each, or accept staleness",
            )
        ]
    return [
        Check(
            "manifest.fresh",
            CheckStatus.OK,
            f"manifest has {len(m.datasets)} datasets, all fresh",
        )
    ]


def run_doctor() -> list[Check]:
    cid = detect_cluster()
    results: list[Check] = []
    results.extend(_check_topology())
    results.extend(_check_ssh(cid))
    results.extend(_check_manifest())
    return results
