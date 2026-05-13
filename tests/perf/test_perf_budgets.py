import time

from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec
from salvo.topology.loader import load_cluster


def test_yaml_load_under_50ms():
    start = time.perf_counter()
    for _ in range(10):
        load_cluster("rorqual")
    elapsed_ms = (time.perf_counter() - start) * 1000 / 10
    assert elapsed_ms < 50, f"yaml load avg {elapsed_ms:.1f}ms exceeds 50ms budget"


def test_render_under_10ms():
    cluster = load_cluster("rorqual")
    spec = JobSpec(name="t", cmd=["echo"], gpus=1, cpus=8, mem="32G", time="2h")
    start = time.perf_counter()
    for _ in range(100):
        render_sbatch(
            spec,
            cluster=cluster,
            account="rrg-bengioy-ad",
            partition="gpubase_bygpu_b1",
            artifact_dir="/tmp/x",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000 / 100
    assert elapsed_ms < 10, f"render avg {elapsed_ms:.1f}ms exceeds 10ms budget"
