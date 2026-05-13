import pytest
from salvo.dispatch.account import pick_account
from salvo.dispatch.caps import CapsSnapshot
from salvo.errors import NoAccountError
from salvo.job.spec import JobSpec
from salvo.topology.loader import load_cluster


def test_gpu_job_prefers_rrg_then_ctb():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=8, mem="16G", time="2h")
    snap = CapsSnapshot()
    assert pick_account(spec, c, snap) == "rrg-bengioy-ad"


def test_cpu_job_forbidden_for_ctb_drolnick():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=0, cpus=8, mem="16G", time="2h")
    snap = CapsSnapshot()
    chosen = pick_account(spec, c, snap)
    assert chosen != "ctb-drolnick"


def test_falls_through_when_rrg_full():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=8, mem="16G", time="2h")
    snap = CapsSnapshot(gpus_in_use={"rrg-bengioy-ad": 32})
    chosen = pick_account(spec, c, snap)
    assert chosen == "ctb-drolnick"


def test_explicit_override():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, account="def-bengioy")
    snap = CapsSnapshot()
    assert pick_account(spec, c, snap) == "def-bengioy"


def test_no_eligible_account():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=64)
    snap = CapsSnapshot()
    with pytest.raises(NoAccountError):
        pick_account(spec, c, snap)
