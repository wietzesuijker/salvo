import pytest
from salvo.job.oom import (
    BumpGpusStep,
    BumpMemStep,
    CallbackStep,
    EscalatePartitionStep,
    FailStep,
    parse_policy,
)


def test_bump_mem():
    p = parse_policy(["bump_mem(1.5x, max=512G)"])
    assert isinstance(p[0], BumpMemStep)
    assert p[0].factor == 1.5
    assert p[0].max_mb == 512 * 1024


def test_full_policy():
    p = parse_policy(
        [
            "bump_mem(2x, max=256G)",
            "escalate_partition",
            "bump_gpus(2x, max=4)",
            "callback(myproject.oom:halve_batch)",
            "fail",
        ]
    )
    assert isinstance(p[0], BumpMemStep)
    assert isinstance(p[1], EscalatePartitionStep)
    assert isinstance(p[2], BumpGpusStep)
    assert isinstance(p[3], CallbackStep)
    assert p[3].target == "myproject.oom:halve_batch"
    assert isinstance(p[4], FailStep)


def test_unknown_step_raises():
    with pytest.raises(ValueError):
        parse_policy(["bogus(arg)"])


def test_empty_defaults_to_fail():
    assert isinstance(parse_policy([])[0], FailStep)


def test_callback_missing_colon_raises():
    with pytest.raises(ValueError):
        parse_policy(["callback(myproject.halve_batch)"])
