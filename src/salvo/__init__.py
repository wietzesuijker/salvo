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
from salvo.errors import SalvoError
from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.job.submit import submit
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster


def render(
    spec: JobSpec,
    *,
    cluster_id: str | None = None,
    artifact_dir: str = "<artifact_dir>",
) -> str:
    """Render a JobSpec to sbatch text. Byte-stable for the same inputs.

    Picks account + partition via salvo.dispatch unless the spec pins them.
    The artifact_dir placeholder is substituted verbatim; callers are
    expected to manage their own artifact paths (or use salvo.submit).
    """
    cid = cluster_id or detect_cluster()
    if cid is None:
        raise SalvoError("could not detect cluster; pass cluster_id= or set SALVO_CLUSTER")
    cluster_obj = load_cluster(cid)
    caps = CapsTracker(cluster_id=cid, user=os.environ.get("USER", "unknown"))
    account = pick_account(spec, cluster_obj, caps.snapshot())
    partition = pick_partition(spec, cluster_obj, account=account)
    return render_sbatch(
        spec,
        cluster=cluster_obj,
        account=account,
        partition=partition,
        artifact_dir=artifact_dir,
    )


__all__ = [
    "JobSpec",
    "PythonEntrypoint",
    "__version__",
    "cluster",
    "render",
    "submit",
]
