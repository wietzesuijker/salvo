"""End-to-end example: how cluv would call salvo as its retry-policy library.

A cluv maintainer should be able to read this file on a laptop and trust
that wiring salvo into cluv's submit flow is a small, well-defined change.
It walks the natural composition described in salvo's README ("Use with
cluv"): parse the on_oom DSL from pyproject, build a JobSpec from values
cluv already has, run apply_oom on a synthetic OOM, and render the bumped
spec to sbatch text. No SLURM, no network, no subprocess — just stdlib +
pydantic. Run with: ``python examples/cluv_integration.py``.
"""

from __future__ import annotations

from salvo import JobSpec, render
from salvo.policy import OomContext, apply_oom, parse


def main() -> None:
    # 1. Parse the policy DSL from a string list. In real cluv this list
    #    comes from `[tool.cluv.retry].on_oom` in the user's pyproject.toml.
    on_oom_dsl: list[str] = ["bump_mem(1.5x, max=128G)", "fail"]
    steps = parse(on_oom_dsl)
    print(f"parsed {len(steps)} policy step(s): {[type(s).__name__ for s in steps]}")

    # 2. Build a minimal JobSpec from values cluv already has. cluv knows
    #    the entrypoint, gpu/cpu ask, env-var-style mem string, and the
    #    user's on_oom list — exactly what JobSpec needs.
    spec = JobSpec(
        name="train",
        cmd=["python", "train.py"],
        gpus=1,
        cpus=8,
        mem="32G",
        time="2h",
        on_oom=on_oom_dsl,
        max_hops=5,
    )

    # 6. Walk all three hops of the bump_mem ladder so the example shows
    #    the policy doing something across resubmissions, not just one call.
    #    32G -> 48G -> 72G -> 108G (1.5x each, capped at 128G).
    current: JobSpec | None = spec
    for hop in range(1, 4):
        assert current is not None  # narrowing for type-checkers
        mem_before = current.mem

        # 3. Synthesize an OOM context. cluv builds this from the job state
        #    it already tracks (sacct max_rss, tail of stderr, cpu/gpu class).
        ctx = OomContext(
            class_="cpu",
            max_rss_mb=current.mem_mb(),  # observed RSS pinned to current ask
            log_excerpt="slurmstepd: error: Detected 1 oom_kill event",
        )

        # 4. Apply the policy. Two outcomes:
        #      (new_spec, "bump_mem")       -> resubmit with new_spec.mem
        #      (None,     "fail")           -> stop, surface to user
        new_spec, action = apply_oom(current, ctx)
        if new_spec is None:
            print(f"hop {hop}: action={action!r}  mem {mem_before} -> (stop)")
            break

        print(f"hop {hop}: action={action!r}  mem {mem_before} -> {new_spec.mem}")
        current = new_spec

    # 5. Render the final bumped spec to sbatch text. With both account and
    #    partition supplied this is pure: no squeue, no SLURM, no I/O. cluv
    #    would write this string to a file and `sbatch` it; here we just
    #    print the header so the audit trail is visible.
    assert current is not None
    sbatch_text = render(
        current,
        cluster_id="mila",
        account="mila",
        partition="unkillable",
    )
    print("\nrendered sbatch (header only):")
    for line in sbatch_text.splitlines()[:8]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
