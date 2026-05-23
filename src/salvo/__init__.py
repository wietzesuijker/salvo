"""salvo public API.

salvo owns policy (OOM retry, account/partition picking) and render
(JobSpec -> byte-stable sbatch text). Transport lives elsewhere (cluv).

Library surface:

    from salvo import JobSpec, render, cluster
    from salvo.policy import parse, apply_oom, Step, OomContext
    from salvo.policy import BumpMemStep, EscalatePartitionStep, FailStep
    from salvo.topology import load_preset, list_presets, ClusterTopology
"""

from __future__ import annotations

import os

from salvo._version import __version__
from salvo.decorators import cluster
from salvo.dispatch.account import pick_account
from salvo.dispatch.caps import CapsTracker
from salvo.dispatch.partition import pick_partition
from salvo.errors import (
    ClusterYAMLError,
    DataNotStagedError,
    DispatchError,
    EventSchemaError,
    ManifestError,
    MaxHopsExceededError,
    NoAccountError,
    NoPartitionError,
    OomPolicyError,
    SalvoError,
    SpecValidationError,
    SubprocessError,
)
from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.job.submit import submit
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster


def render(
    spec: JobSpec,
    *,
    cluster_id: str | None = None,
    account: str | None = None,
    partition: str | None = None,
    artifact_dir: str = "<artifact_dir>",
) -> str:
    """Render a JobSpec to sbatch text. Byte-stable for the same inputs.

    Pure when ``account`` and ``partition`` are provided (or pinned on the
    spec). Library callers (cluv, xgenius, etc.) should pass them explicitly
    so render() has no side effects. When either is missing, salvo.dispatch
    fills it in by shelling out to squeue for live capacity; that path only
    works on a SLURM login node.
    """
    cid = cluster_id or detect_cluster()
    if cid is None:
        raise SalvoError("could not detect cluster; pass cluster_id= or set SALVO_CLUSTER")
    cluster_obj = load_cluster(cid)
    acct = account or spec.account
    part = partition or spec.partition
    if acct is None or part is None:
        caps = CapsTracker(cluster_id=cid, user=os.environ.get("USER", "unknown"))
        snap = caps.snapshot()
        acct = acct or pick_account(spec, cluster_obj, snap)
        part = part or pick_partition(spec, cluster_obj, account=acct)
    return render_sbatch(
        spec,
        cluster=cluster_obj,
        account=acct,
        partition=part,
        artifact_dir=artifact_dir,
    )


__all__ = [
    "ClusterYAMLError",
    "DataNotStagedError",
    "DispatchError",
    "EventSchemaError",
    "JobSpec",
    "ManifestError",
    "MaxHopsExceededError",
    "NoAccountError",
    "NoPartitionError",
    "OomPolicyError",
    "PythonEntrypoint",
    "SalvoError",
    "SpecValidationError",
    "SubprocessError",
    "__version__",
    "cluster",
    "render",
    "submit",
]
