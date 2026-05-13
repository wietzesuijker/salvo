"""JobSpec + Cluster → sbatch.sh. Byte-identical for same inputs (spec §4.4 invariant 5)."""

from __future__ import annotations

import json
import shlex

from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.topology.schema import Cluster

_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_mb}M
#SBATCH --time={time_min}
{gres_line}{output_line}{error_line}{signal_line}
set -euo pipefail
export SALVO_ARTIFACT_DIR={artifact_dir_quoted}
export SALVO_HOP=${{SALVO_HOP:-0}}/{max_hops}
{env_lines}{container_prefix}{cmd_line}
"""


def _format_cmd(cmd: list[str] | PythonEntrypoint) -> str:
    if isinstance(cmd, PythonEntrypoint):
        payload = json.dumps({"target": cmd.target, "kwargs": cmd.kwargs}, sort_keys=True)
        return f"python -m salvo.runtime.entrypoint {shlex.quote(payload)}"
    return " ".join(shlex.quote(a) for a in cmd)


def render_sbatch(
    spec: JobSpec,
    *,
    cluster: Cluster,
    account: str,
    partition: str,
    artifact_dir: str,
) -> str:
    gres_line = f"#SBATCH --gres=gpu:{spec.gpus}\n" if spec.gpus > 0 else ""
    out_path = f"{artifact_dir}/stdout.log"
    err_path = f"{artifact_dir}/stderr.log"
    output_line = f"#SBATCH --output={out_path}\n"
    error_line = f"#SBATCH --error={err_path}\n"
    signal_line = "#SBATCH --signal=SIGUSR1@90\n" if spec.on_preempt == "resubmit" else ""
    env_lines = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(spec.env.items()))
    container_prefix = ""
    if spec.container:
        container_prefix = f"apptainer exec --nv {spec.container} \\\n  "
    return _TEMPLATE.format(
        name=spec.name,
        account=account,
        partition=partition,
        cpus=spec.cpus,
        mem_mb=spec.mem_mb(),
        time_min=spec.time_min(),
        gres_line=gres_line,
        output_line=output_line,
        error_line=error_line,
        signal_line=signal_line,
        artifact_dir_quoted=shlex.quote(artifact_dir),
        max_hops=spec.max_hops,
        env_lines=env_lines,
        container_prefix=container_prefix,
        cmd_line=_format_cmd(spec.cmd),
    )
