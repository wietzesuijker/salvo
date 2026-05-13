"""End-to-end submit pipeline."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from salvo._version import __version__
from salvo.dispatch.account import pick_account
from salvo.dispatch.caps import CapsTracker
from salvo.dispatch.partition import pick_partition
from salvo.errors import SalvoError
from salvo.job.handle import JobHandle
from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec
from salvo.manifest.store import Manifest
from salvo.obs.events import EventEmitter
from salvo.stage.gate import assert_data_available
from salvo.topology.detect import detect_cluster
from salvo.topology.loader import load_cluster

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def submit(
    spec: JobSpec,
    *,
    cluster_id: str | None = None,
    salvo_root: Path | None = None,
    allow_missing_data: bool = False,
) -> JobHandle:
    cluster_id = cluster_id or detect_cluster()
    if cluster_id is None:
        raise SalvoError("could not detect cluster; set SALVO_CLUSTER or pass cluster_id=")
    cluster = load_cluster(cluster_id)
    salvo_root = salvo_root or Path(os.path.expanduser("~/.salvo"))
    manifest = Manifest.load(salvo_root / "manifest.toml")

    user = os.environ.get("USER", "unknown")
    caps = CapsTracker(cluster_id=cluster_id, user=user)

    pending_id = f"pending-{os.getpid()}-{int.from_bytes(os.urandom(4), 'big'):08x}"
    artifact_dir = salvo_root / "runs" / pending_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    emitter = EventEmitter(artifact_dir / "events.jsonl")

    try:
        emitter.emit(
            "submit.attempt", cluster=cluster_id, name=spec.name, salvo_version=__version__
        )
        assert_data_available(
            spec, cluster_id=cluster_id, manifest=manifest, allow_missing=allow_missing_data
        )
        account = pick_account(spec, cluster, caps.snapshot())
        emitter.emit("dispatch.account_picked", account=account, cluster=cluster_id, name=spec.name)
        partition = pick_partition(spec, cluster, account=account)
        emitter.emit(
            "dispatch.partition_picked",
            partition=partition,
            account=account,
            cluster=cluster_id,
        )
    except Exception as e:
        emitter.emit("submit.error", error=str(e), cluster=cluster_id, name=spec.name)
        raise

    script = render_sbatch(
        spec, cluster=cluster, account=account, partition=partition, artifact_dir=str(artifact_dir)
    )
    (artifact_dir / "sbatch.sh").write_text(script)
    (artifact_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
    (artifact_dir / "cluster.json").write_text(cluster.model_dump_json(indent=2))

    proc = subprocess.run(
        ["sbatch", str(artifact_dir / "sbatch.sh")], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        emitter.emit("submit.error", error=proc.stderr.strip(), cluster=cluster_id, name=spec.name)
        raise SalvoError(f"sbatch failed: {proc.stderr.strip()}")
    m = _JOBID_RE.search(proc.stdout)
    if not m:
        emitter.emit(
            "submit.error",
            error=f"unparseable sbatch output: {proc.stdout!r}",
            cluster=cluster_id,
            name=spec.name,
        )
        raise SalvoError(f"could not parse sbatch output: {proc.stdout!r}")
    job_id = m.group(1)

    final_dir = salvo_root / "runs" / job_id
    artifact_dir.rename(final_dir)
    emitter = EventEmitter(final_dir / "events.jsonl")
    emitter.emit(
        "submit.success",
        job_id=job_id,
        cluster=cluster_id,
        account=account,
        partition=partition,
        gpus=spec.gpus,
        mem_mb=spec.mem_mb(),
        time_min=spec.time_min(),
        salvo_version=__version__,
    )
    caps.account_after_submit(account, gpus=spec.gpus, cpus=spec.cpus, mem_mb=spec.mem_mb())
    return JobHandle(job_id=job_id, cluster=cluster_id, spec=spec, artifact_dir=final_dir)
