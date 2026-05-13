import pytest
from salvo.dispatch.partition import pick_partition
from salvo.errors import NoPartitionError
from salvo.job.spec import JobSpec
from salvo.topology.loader import load_cluster


def test_gpu_short_picks_b1():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=8, mem="16G", time="2h")
    assert pick_partition(spec, c, account="rrg-bengioy-ad") == "gpubase_bygpu_b1"


def test_gpu_long_picks_b3():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=8, mem="16G", time="8h")
    assert pick_partition(spec, c, account="rrg-bengioy-ad") == "gpubase_bygpu_b3"


def test_mila_short_gpu_prefers_main_not_long():
    c = load_cluster("mila")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=4, mem="16G", time="2h")
    assert pick_partition(spec, c, account="mila") == "main"


def test_mila_cpu_181g_forbids_main_cpu():
    c = load_cluster("mila")
    spec = JobSpec(name="t", cmd=["echo"], gpus=0, cpus=8, mem="200G", time="1h")
    chosen = pick_partition(spec, c, account="mila")
    assert chosen == "long-cpu"


def test_explicit_override():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], partition="gpubase_bygpu_b3")
    assert pick_partition(spec, c, account="rrg-bengioy-ad") == "gpubase_bygpu_b3"


def test_no_fit_raises():
    c = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, time="48h")
    with pytest.raises(NoPartitionError):
        pick_partition(spec, c, account="rrg-bengioy-ad")
