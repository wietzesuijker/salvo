"""Smoke tests for the public surface: salvo.render, salvo.policy, salvo.topology."""

from __future__ import annotations

import salvo
from salvo import JobSpec, render
from salvo.policy import (
    BumpMemStep,
    EscalatePartitionStep,
    FailStep,
    OomContext,
    OomDecision,
    Step,
    apply_oom,
    parse,
)
from salvo.topology import ClusterTopology, list_presets, load_preset


def test_top_level_exports():
    assert salvo.JobSpec is JobSpec
    assert callable(salvo.render)
    assert callable(salvo.submit)
    assert salvo.cluster is salvo.cluster  # decorator singleton


def test_render_returns_sbatch_text(fake_sbatch, tmp_home):
    spec = JobSpec(name="t", cmd=["echo", "hi"], cpus=2, mem="4G", time="30m")
    out = render(spec, cluster_id="mila", artifact_dir="/tmp/x")
    assert "#SBATCH --job-name=t" in out
    assert "--cpus-per-task=2" in out


def test_render_is_pure_when_account_and_partition_supplied(monkeypatch):
    """Library callers (cluv, xgenius) must be able to render off a SLURM node.

    No subprocess.run mock here on purpose: if render() shells out to squeue
    or anything else when account + partition are explicit, this test will
    raise FileNotFoundError on a non-SLURM machine.
    """

    def _no_subprocess(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError(f"render() shelled out: {args!r} {kwargs!r}")

    monkeypatch.setattr("subprocess.run", _no_subprocess)
    spec = JobSpec(
        name="train",
        cmd=["python", "train.py"],
        gpus=1,
        cpus=8,
        mem="32G",
        time="2h",
        on_oom=["bump_mem(1.5x, max=128G)", "fail"],
    )
    out = render(spec, cluster_id="mila", account="mila", partition="unkillable")
    assert "#SBATCH --account=mila" in out
    assert "#SBATCH --partition=unkillable" in out
    assert "#SBATCH --gres=gpu:a100:1" in out


def test_render_uses_spec_pinned_account_and_partition(monkeypatch):
    """spec.account / spec.partition also avoid the squeue path."""

    def _no_subprocess(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError(f"render() shelled out: {args!r} {kwargs!r}")

    monkeypatch.setattr("subprocess.run", _no_subprocess)
    spec = JobSpec(
        name="t",
        cmd=["echo"],
        cpus=2,
        mem="4G",
        time="30m",
        account="rrg-foo",
        partition="long",
    )
    out = render(spec, cluster_id="mila")
    assert "#SBATCH --account=rrg-foo" in out
    assert "#SBATCH --partition=long" in out


def test_policy_parse_maps_to_steps():
    p = parse(["bump_mem(1.5x, max=128G)", "escalate_partition", "fail"])
    assert isinstance(p[0], BumpMemStep)
    assert isinstance(p[1], EscalatePartitionStep)
    assert isinstance(p[2], FailStep)
    # Step is the union alias
    assert all(isinstance(s, Step) for s in p)


def test_policy_apply_oom_bumps_mem():
    spec = JobSpec(
        name="t",
        cmd=["echo"],
        mem="16G",
        on_oom=["bump_mem(2x, max=128G)", "fail"],
    )
    decision = apply_oom(spec, OomContext(kind="cpu", max_rss_mb=14000))
    # NamedTuple: both attribute access and tuple-unpack must work.
    assert isinstance(decision, OomDecision)
    assert decision.new_spec is not None
    assert decision.action == "bump_mem"
    new_spec, action = decision
    assert new_spec.mem_mb() >= 32 * 1024
    assert action == "bump_mem"


def test_topology_load_preset_and_alias():
    c = load_preset("mila")
    assert isinstance(c, ClusterTopology)
    assert c.id == "mila"


def test_topology_list_presets_excludes_private():
    presets = list_presets()
    assert "mila" in presets
    assert "rorqual" in presets
    assert all(not p.startswith("_") for p in presets)
