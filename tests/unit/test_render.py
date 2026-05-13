from pathlib import Path

from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.topology.loader import load_cluster

GOLDEN = Path(__file__).parent.parent / "integration" / "golden"


def test_mila_cpu_basic_golden():
    spec = JobSpec(
        name="cpu_basic",
        cmd=["python", "-c", "print(1)"],
        gpus=0,
        cpus=4,
        mem="16G",
        time="1h",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="main-cpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert out == GOLDEN.joinpath("mila_cpu_basic.sh").read_text()


def test_rorqual_gpu_basic_golden():
    spec = JobSpec(
        name="gpu_basic",
        cmd=["python", "train.py", "--seed", "1"],
        gpus=1,
        cpus=16,
        mem="32G",
        time="2h",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("rorqual"),
        account="rrg-bengioy-ad",
        partition="gpubase_bygpu_b1",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert out == GOLDEN.joinpath("rorqual_gpu_basic.sh").read_text()


def test_python_entrypoint_renders_uv_run():
    spec = JobSpec(
        name="py",
        cmd=PythonEntrypoint(target="mymod.train:main", kwargs={"seed": 7}),
        gpus=0,
        cpus=1,
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="main-cpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "python -m salvo.runtime.entrypoint" in out
    assert "mymod.train:main" in out
