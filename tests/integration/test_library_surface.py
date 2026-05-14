"""Integration: the README + examples library surface must work off-cluster.

These tests intentionally do NOT use ``fake_sbatch``. They simulate what a
downstream library (cluv, xgenius) sees on a developer laptop with no
SLURM tooling installed: any subprocess.run call is a regression.
"""

from __future__ import annotations

import subprocess

import pytest


def _no_subprocess(*args, **kwargs):  # pragma: no cover - assertion path
    raise AssertionError(f"library surface shelled out: {args!r} {kwargs!r}")


@pytest.fixture
def forbid_subprocess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _no_subprocess)


def test_readme_render_example(forbid_subprocess):
    """The exact code block from README.md must render off-cluster."""
    from salvo import JobSpec, render

    spec = JobSpec(
        name="train",
        cmd=["python", "train.py"],
        gpus=1,
        cpus=8,
        mem="32G",
        time="2h",
        on_oom=["bump_mem(1.5x, max=128G)", "fail"],
    )
    sbatch_text = render(
        spec,
        cluster_id="mila",
        account="mila",
        partition="unkillable",
    )
    assert sbatch_text.startswith("#!/bin/bash\n")
    assert "#SBATCH --job-name=train" in sbatch_text
    assert "#SBATCH --account=mila" in sbatch_text
    assert "#SBATCH --partition=unkillable" in sbatch_text
    assert "#SBATCH --gres=gpu:1" in sbatch_text


def test_readme_policy_example(forbid_subprocess):
    """The OOM policy code block from README.md must apply off-cluster."""
    from salvo import JobSpec
    from salvo.policy import OomContext, apply_oom, parse

    steps = parse(["bump_mem(1.5x, max=128G)", "escalate_partition", "fail"])
    assert len(steps) == 3

    prev_spec = JobSpec(
        name="t",
        cmd=["echo"],
        mem="16G",
        on_oom=["bump_mem(1.5x, max=128G)", "escalate_partition", "fail"],
    )
    new_spec, action = apply_oom(prev_spec, OomContext(class_="cpu", max_rss_mb=15500))
    assert new_spec is not None
    assert action == "bump_mem"
    assert new_spec.mem_mb() > prev_spec.mem_mb()


def test_readme_topology_example(forbid_subprocess):
    """The topology code block from README.md must list and load off-cluster."""
    from salvo.topology import ClusterTopology, list_presets, load_preset

    presets = list_presets()
    assert presets == sorted(presets)
    assert "mila" in presets
    cluster = load_preset("mila")
    assert isinstance(cluster, ClusterTopology)
    assert cluster.id == "mila"


def test_byte_stable_render_is_idempotent(forbid_subprocess):
    """Same JobSpec + same args -> identical bytes. The README's audit promise."""
    from salvo import JobSpec, render

    spec = JobSpec(
        name="audit",
        cmd=["python", "x.py"],
        gpus=1,
        cpus=4,
        mem="8G",
        time="1h",
    )
    a = render(spec, cluster_id="mila", account="mila", partition="unkillable")
    b = render(spec, cluster_id="mila", account="mila", partition="unkillable")
    assert a == b
