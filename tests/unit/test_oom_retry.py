from salvo.job.oom import OomContext, apply_oom_policy
from salvo.job.spec import JobSpec


def test_bump_mem_returns_new_spec():
    spec = JobSpec(name="t", cmd=["echo"], mem="16G", on_oom=["bump_mem(2x, max=128G)", "fail"])
    ctx = OomContext(class_="cpu", max_rss_mb=14000)
    new_spec, action = apply_oom_policy(spec, ctx)
    assert new_spec is not None
    assert new_spec.mem_mb() >= 32 * 1024
    assert new_spec.mem_mb() <= 128 * 1024
    assert action == "bump_mem"


def test_bump_mem_capped():
    spec = JobSpec(name="t", cmd=["echo"], mem="100G", on_oom=["bump_mem(5x, max=128G)", "fail"])
    new_spec, _action = apply_oom_policy(spec, OomContext(class_="cpu", max_rss_mb=90000))
    assert new_spec is not None
    assert new_spec.mem_mb() == 128 * 1024


def test_gpu_oom_callback_not_applied_phase0():
    spec = JobSpec(name="t", cmd=["echo"], on_oom=["callback(my.mod:fn)", "fail"])
    new_spec, action = apply_oom_policy(spec, OomContext(class_="gpu"))
    assert new_spec is None
    assert action == "fail"
