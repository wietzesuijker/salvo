"""JobSpec + Cluster → sbatch.sh. Byte-identical for same inputs (spec §4.4 invariant 5)."""

from __future__ import annotations

import json
import shlex

from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.topology.schema import Cluster, Partition

_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
{mem_line}#SBATCH --time={time_min}
#SBATCH --export=NONE
{qos_line}{gres_line}{output_line}{error_line}{signal_line}
set -euo pipefail
export SALVO_ARTIFACT_DIR={artifact_dir_quoted}
export SALVO_HOP=${{SALVO_HOP:-0}}/{max_hops}
{trap_line}{env_lines}{container_prefix}{cmd_line}
"""


def _format_cmd(cmd: list[str] | PythonEntrypoint) -> str:
    if isinstance(cmd, PythonEntrypoint):
        payload = json.dumps({"target": cmd.target, "kwargs": cmd.kwargs}, sort_keys=True)
        return f"python -m salvo.runtime.entrypoint {shlex.quote(payload)}"
    return " ".join(shlex.quote(a) for a in cmd)


def _lookup_partition(cluster: Cluster, name: str) -> Partition | None:
    for p in cluster.partitions:
        if p.name == name:
            return p
    return None


def render_sbatch(
    spec: JobSpec,
    *,
    cluster: Cluster,
    account: str,
    partition: str,
    artifact_dir: str,
) -> str:
    part = _lookup_partition(cluster, partition)
    mem_mb = spec.mem_mb()
    if part is not None and part.prefer_mem_kind == "per-gpu" and spec.gpus > 0:
        mem_line = f"#SBATCH --mem-per-gpu={mem_mb // spec.gpus}M\n"
    else:
        mem_line = f"#SBATCH --mem={mem_mb}M\n"
    qos_line = (
        f"#SBATCH --qos={part.requires_qos}\n" if part is not None and part.requires_qos else ""
    )
    if spec.gpus > 0:
        if cluster.gpu_model:
            gres_line = f"#SBATCH --gres=gpu:{cluster.gpu_model}:{spec.gpus}\n"
        else:
            gres_line = f"#SBATCH --gres=gpu:{spec.gpus}\n"
    else:
        gres_line = ""
    out_path = f"{artifact_dir}/stdout.log"
    err_path = f"{artifact_dir}/stderr.log"
    output_line = f"#SBATCH --output={out_path}\n"
    error_line = f"#SBATCH --error={err_path}\n"
    signal_line = "#SBATCH --signal=SIGUSR1@90\n" if spec.on_preempt == "resubmit" else ""
    if signal_line:
        trap_line = (
            'if [[ "${SLURM_JOB_ID:-}" != "" ]]; then\n'
            "  trap 'python -m salvo.runtime.epilog "
            '--artifact-dir "$SALVO_ARTIFACT_DIR" --reason preempt\' SIGUSR1\n'
            "fi\n"
        )
    else:
        trap_line = ""
    env_lines = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(spec.env.items()))
    container_prefix = ""
    if spec.container:
        container_prefix = f"apptainer exec --nv {spec.container} \\\n  "
    return _TEMPLATE.format(
        name=spec.name,
        account=account,
        partition=partition,
        cpus=spec.cpus,
        mem_line=mem_line,
        time_min=spec.time_min(),
        qos_line=qos_line,
        gres_line=gres_line,
        output_line=output_line,
        error_line=error_line,
        signal_line=signal_line,
        trap_line=trap_line,
        artifact_dir_quoted=shlex.quote(artifact_dir),
        max_hops=spec.max_hops,
        env_lines=env_lines,
        container_prefix=container_prefix,
        cmd_line=_format_cmd(spec.cmd),
    )
