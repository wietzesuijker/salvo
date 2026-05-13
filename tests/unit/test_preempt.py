from salvo.job.preempt import next_hop, should_resubmit, strip_account_suffix


def test_strip_drac_suffix():
    assert strip_account_suffix("rrg-bengioy-ad_gpu") == "rrg-bengioy-ad"
    assert strip_account_suffix("rrg-bengioy-ad_cpu") == "rrg-bengioy-ad"
    assert strip_account_suffix("mila") == "mila"


def test_next_hop_increments():
    assert next_hop("0/5") == ("1/5", False)
    assert next_hop("4/5") == ("5/5", False)
    assert next_hop("5/5") == ("5/5", True)


def test_should_resubmit_within_budget():
    assert should_resubmit(hop_env="0/5", on_preempt="resubmit", artifact_changed=True) is True


def test_should_resubmit_no_progress():
    assert should_resubmit(hop_env="0/5", on_preempt="resubmit", artifact_changed=False) is False


def test_should_resubmit_fail_policy():
    assert should_resubmit(hop_env="0/5", on_preempt="fail", artifact_changed=True) is False


def test_should_resubmit_max_reached():
    assert should_resubmit(hop_env="5/5", on_preempt="resubmit", artifact_changed=True) is False
