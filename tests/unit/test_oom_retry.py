from salvo.job.oom import OomContext, apply_oom_policy
from salvo.job.spec import JobSpec


def test_bump_mem_returns_new_spec():
    spec = JobSpec(name="t", cmd=["echo"], mem="16G", on_oom=["bump_mem(2x, max=128G)", "fail"])
    ctx = OomContext(kind="cpu", max_rss_mb=14000)
    new_spec, action = apply_oom_policy(spec, ctx)
    assert new_spec is not None
    assert new_spec.mem_mb() >= 32 * 1024
    assert new_spec.mem_mb() <= 128 * 1024
    assert action == "bump_mem"


def test_bump_mem_capped():
    spec = JobSpec(name="t", cmd=["echo"], mem="100G", on_oom=["bump_mem(5x, max=128G)", "fail"])
    new_spec, _action = apply_oom_policy(spec, OomContext(kind="cpu", max_rss_mb=90000))
    assert new_spec is not None
    assert new_spec.mem_mb() == 128 * 1024


def test_gpu_oom_callback_not_applied_phase0():
    spec = JobSpec(name="t", cmd=["echo"], on_oom=["callback(my.mod:fn)", "fail"])
    new_spec, action = apply_oom_policy(spec, OomContext(kind="gpu"))
    assert new_spec is None
    assert action == "fail"


def test_bump_mem_skipped_when_kind_is_gpu():
    # GPU-side OOM should not bump host memory; bump_mem falls through and
    # the next step decides. With only [bump_mem, fail] the result is fail.
    spec = JobSpec(name="t", cmd=["echo"], mem="16G", on_oom=["bump_mem(2x, max=128G)", "fail"])
    new_spec, action = apply_oom_policy(spec, OomContext(kind="gpu", max_rss_mb=14000))
    assert new_spec is None
    assert action == "fail"


def test_bump_mem_skipped_on_gpu_then_escalate_partition_fires():
    # Confirms the fall-through composition: gpu OOM skips bump_mem, then
    # escalate_partition picks up the slack by clearing partition.
    spec = JobSpec(
        name="t",
        cmd=["echo"],
        mem="16G",
        partition="main",
        on_oom=["bump_mem(2x, max=128G)", "escalate_partition", "fail"],
    )
    new_spec, action = apply_oom_policy(spec, OomContext(kind="gpu"))
    assert new_spec is not None
    assert new_spec.partition is None
    assert action == "escalate_partition"
