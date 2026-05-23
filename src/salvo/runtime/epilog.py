"""SIGUSR1 trap handler. Resubmits the running job before SLURM preempts it.

Invoked from the rendered sbatch script's ``trap`` clause when ``spec.on_preempt
== "resubmit"``:

.. code-block:: bash

    python -m salvo.runtime.epilog --artifact-dir "$SALVO_ARTIFACT_DIR" \\
        --reason preempt

Reads the parent's ``spec.json`` + ``cluster.json``, bumps ``SALVO_HOP``, calls
``salvo.job.submit.submit`` to enqueue a child job, and records the transition
as a ``Hop`` entry in the parent's ``hops.jsonl``. On error: log to events.jsonl
and exit 1 so the bash trap fails loudly.

Imports ``salvo.*`` because the wrapper runs in the same env as the user's job,
which must have ``pysalvo`` installed to render this script in the first place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from salvo.errors import SalvoError
from salvo.job.handle import JobHandle
from salvo.job.preempt import next_hop
from salvo.job.spec import Hop, JobSpec
from salvo.obs.events import EventEmitter

Reason = Literal["preempt", "oom", "manual"]


def _submit(
    spec: JobSpec,
    *,
    cluster_id: str,
    hop: str,
    parent_artifact_dir: Path,
) -> JobHandle:
    """Thin indirection over ``salvo.job.submit.submit`` so tests can patch it.

    Imported lazily to keep epilog importable on broken installs (the trap
    runs from inside a live job; a top-level import failure would explode the
    whole process and we'd lose the chance to emit a clean error event).
    """
    from salvo.job.submit import submit as _real_submit

    return _real_submit(
        spec,
        cluster_id=cluster_id,
        hop=hop,
        parent_artifact_dir=parent_artifact_dir,
    )


def _read_parent_job_id(artifact_dir: Path) -> str | None:
    """Best-effort: find ``job_id`` from the submit.success event line."""
    events_file = artifact_dir / "events.jsonl"
    if not events_file.exists():
        return None
    for line in events_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "submit.success" and "job_id" in rec:
            jid = rec["job_id"]
            return str(jid) if jid is not None else None
    return None


def run(*, artifact_dir: Path, reason: Reason) -> int:
    """Core handler. Returns process exit code (0 ok, 1 error)."""
    if not artifact_dir.exists() or not artifact_dir.is_dir():
        # Can't even emit an event if the dir is missing. Log to stderr.
        print(
            f"salvo.runtime.epilog: artifact dir does not exist: {artifact_dir}",
            file=sys.stderr,
        )
        return 1

    emitter = EventEmitter(artifact_dir / "events.jsonl")

    try:
        spec_path = artifact_dir / "spec.json"
        if not spec_path.exists():
            emitter.emit(
                "submit.error",
                error=f"epilog: spec.json missing at {spec_path}",
                reason=reason,
            )
            return 1
        spec = JobSpec.model_validate_json(spec_path.read_text())

        # Fall back to spec.max_hops if SALVO_HOP didn't propagate (e.g.
        # --export=NONE blocked env inheritance on a resubmit chain).
        hop_env = os.environ.get("SALVO_HOP", f"0/{spec.max_hops}")

        new_hop, max_exceeded = next_hop(hop_env)
        parent_job_id = _read_parent_job_id(artifact_dir) or "unknown"

        if max_exceeded:
            emitter.emit(
                "hop.max_exceeded",
                parent_job_id=parent_job_id,
                hop=hop_env,
                reason=reason,
            )
            return 0

        cluster_path = artifact_dir / "cluster.json"
        if not cluster_path.exists():
            emitter.emit(
                "submit.error",
                error=f"epilog: cluster.json missing at {cluster_path}",
                reason=reason,
            )
            return 1
        cluster_blob = json.loads(cluster_path.read_text())
        cluster_id = cluster_blob.get("id")
        if not isinstance(cluster_id, str) or not cluster_id:
            emitter.emit(
                "submit.error",
                error=f"epilog: cluster.json missing 'id' field: {cluster_blob!r}",
                reason=reason,
            )
            return 1

        try:
            child = _submit(
                spec,
                cluster_id=cluster_id,
                hop=new_hop,
                parent_artifact_dir=artifact_dir,
            )
        except SalvoError as exc:
            emitter.emit("submit.error", error=str(exc), reason=reason)
            return 1

        # Record the hop transition in the parent's hops.jsonl.
        hop_index = int(new_hop.split("/")[0])
        cause: Literal["preempt", "oom_cpu", "oom_gpu", "max_hops_exceeded"]
        cause = "preempt" if reason == "preempt" else "oom_cpu"
        hop_record = Hop(
            parent_job_id=parent_job_id,
            child_job_id=child.job_id,
            cause=cause,
            hop_index=hop_index,
        )
        hops_file = artifact_dir / "hops.jsonl"
        with hops_file.open("a") as f:
            f.write(hop_record.model_dump_json() + "\n")

        emitter.emit(
            "hop.preempt" if reason == "preempt" else "hop.oom",
            parent_job_id=parent_job_id,
            child_job_id=child.job_id,
            hop=new_hop,
        )
        return 0
    except Exception as exc:  # pragma: no cover - defensive
        emitter.emit("submit.error", error=f"epilog: unexpected: {exc}", reason=reason)
        return 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="salvo.runtime.epilog")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--reason", required=True, choices=("preempt", "oom", "manual"))
    args = parser.parse_args(argv)
    rc = run(artifact_dir=args.artifact_dir, reason=args.reason)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
