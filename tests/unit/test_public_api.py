"""Smoke tests for the public surface: salvo.render, salvo.policy, salvo.topology."""

from __future__ import annotations

import salvo
from salvo import JobSpec, render
from salvo.policy import (
    BumpMemStep,
    EscalatePartitionStep,
    FailStep,
    OomContext,
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
    new_spec, action = apply_oom(spec, OomContext(class_="cpu", max_rss_mb=14000))
    assert new_spec is not None
    assert action == "bump_mem"
    assert new_spec.mem_mb() >= 32 * 1024


def test_topology_load_preset_and_alias():
    c = load_preset("mila")
    assert isinstance(c, ClusterTopology)
    assert c.id == "mila"


def test_topology_list_presets_excludes_private():
    presets = list_presets()
    assert "mila" in presets
    assert "rorqual" in presets
    assert all(not p.startswith("_") for p in presets)
