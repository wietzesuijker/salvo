"""End-to-end example: how xgenius would call salvo as its OOM-policy library.

xgenius already has its own sbatch renderer (Jinja-style ``{{PLACEHOLDER}}``
templates in ``xgenius/templates.py``) and its own SafetyValidator with a
fixed ``max_memory_per_job`` ceiling. It has no dynamic OOM handling: an
OOM-killed job is observed by the watcher daemon but no corrective action
is taken. salvo.policy fills exactly that gap. This example shows the
policy-only integration path: xgenius keeps its template renderer; only
the watcher imports salvo. No SLURM, no network, no subprocess — just
stdlib + pydantic. Run with: ``python examples/xgenius_integration.py``.
"""

from __future__ import annotations

from salvo import JobSpec
from salvo.policy import OomContext, OomDecision, apply_oom, parse


def main() -> None:
    # 1. Parse the policy from a future xgenius config knob, e.g. a new
    #    ``[retry] on_oom = [...]`` table in ``xgenius.toml``. Validation
    #    happens here: an unknown step raises ValueError pointing at the
    #    bad string, before any job runs.
    on_oom_dsl: list[str] = ["bump_mem(2x, max=80G)", "fail"]
    steps = parse(on_oom_dsl)
    print(f"parsed {len(steps)} policy step(s): {[type(s).__name__ for s in steps]}")

    # 2. The xgenius watcher detects OUT_OF_MEMORY for an experiment.
    #    It already knows the experiment's gpu/cpu/mem/walltime ask
    #    (from xgenius.toml + the running job record). Wrap that in a
    #    JobSpec — JobSpec is the only salvo type the watcher needs to
    #    build. The cmd field is unused by apply_oom; pass a sentinel.
    experiment_gpus = 1
    experiment_cpus = 4
    experiment_mem = "16G"
    experiment_walltime = "4h"

    spec = JobSpec(
        name="experiment-42",
        cmd=["xgenius-runner"],
        gpus=experiment_gpus,
        cpus=experiment_cpus,
        mem=experiment_mem,
        time=experiment_walltime,
        on_oom=on_oom_dsl,
    )

    # 3. Synthesize an OomContext from what the watcher already polls.
    #    xgenius's COMPLETION_EPILOG drops a trap-based marker on OOM;
    #    MaxRSS is read via sacct the same way cluv does. ``kind`` is
    #    "gpu" only when the job asked for gpus AND the OOM came from
    #    GPU memory exhaustion. For CPU-side OOM on a GPU job, "cpu" is
    #    still the right kind because that is the resource being bumped.
    observed_max_rss_mb = 15_900
    ctx = OomContext(
        kind="cpu",
        max_rss_mb=observed_max_rss_mb,
        log_excerpt="slurmstepd: error: Detected 1 oom_kill event",
    )

    # 4. Ask salvo what to do. The return is a NamedTuple — both
    #    attribute and tuple-unpack work, so the watcher can pick
    #    whichever style reads cleaner in its existing code.
    decision: OomDecision = apply_oom(spec, ctx)
    if decision.new_spec is None:
        print(f"policy terminated: action={decision.action!r} — escalate to human")
        return

    # 5. xgenius now re-runs its own template renderer with the bumped
    #    memory string. salvo does not render xgenius templates; the
    #    watcher just substitutes ``new_spec.mem`` back into the params
    #    dict it already builds in ``xgenius/jobs.py:submit``.
    new_params = {
        "memory": decision.new_spec.mem,  # "32768M" after 16G * 2
        "gpus": decision.new_spec.gpus,
        "cpus": decision.new_spec.cpus,
        "walltime": decision.new_spec.time,
    }
    print(f"action={decision.action!r}")
    print(f"  mem  {experiment_mem} -> {decision.new_spec.mem}")
    print("  hand back to xgenius template renderer with params:")
    for k, v in new_params.items():
        print(f"    {k}={v!r}")


if __name__ == "__main__":
    main()
