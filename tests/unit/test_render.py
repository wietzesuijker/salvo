from pathlib import Path

from salvo.job.render import render_sbatch
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.topology.loader import load_cluster
from salvo.topology.schema import Account, Cluster, Partition

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


def test_render_drac_gpu_uses_mem_per_gpu():
    spec = JobSpec(
        name="gpu",
        cmd=["python", "train.py"],
        gpus=2,
        cpus=32,
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
    assert "#SBATCH --mem-per-gpu=16384M" in out
    assert "#SBATCH --mem=" not in out


def test_render_drac_cpu_uses_total_mem():
    spec = JobSpec(
        name="cpu",
        cmd=["python", "x.py"],
        gpus=0,
        cpus=4,
        mem="16G",
        time="1h",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("rorqual"),
        account="rrg-bengioy-ad",
        partition="cpubase_bycore_b1",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "#SBATCH --mem=16384M" in out
    assert "--mem-per-gpu" not in out


def test_render_mila_long_emits_qos():
    spec = JobSpec(
        name="lcpu",
        cmd=["python", "x.py"],
        gpus=0,
        cpus=4,
        mem="32G",
        time="10h",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="long",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "#SBATCH --qos=long" in out


def test_render_main_partition_no_qos():
    spec = JobSpec(
        name="cpu",
        cmd=["python", "x.py"],
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
    assert "--qos" not in out


def test_render_always_exports_none():
    spec = JobSpec(
        name="x",
        cmd=["python", "x.py"],
        gpus=0,
        cpus=1,
        mem="4G",
        time="30m",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="main-cpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "#SBATCH --export=NONE" in out


def test_render_drac_gpu_model_qualifier():
    spec = JobSpec(
        name="gpu",
        cmd=["python", "train.py"],
        gpus=2,
        cpus=32,
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
    assert "#SBATCH --gres=gpu:h100:2" in out


def test_render_with_preempt_emits_trap():
    """on_preempt='resubmit' must emit a SIGUSR1 trap that calls salvo.runtime.epilog."""
    spec = JobSpec(
        name="trapped",
        cmd=["python", "x.py"],
        gpus=0,
        cpus=1,
        mem="4G",
        time="30m",
        on_preempt="resubmit",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="main-cpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "trap " in out
    assert "salvo.runtime.epilog" in out
    assert "SIGUSR1" in out


def test_render_without_preempt_no_trap():
    """on_preempt='fail' must NOT emit a SIGUSR1 trap (nothing to handle)."""
    spec = JobSpec(
        name="untrapped",
        cmd=["python", "x.py"],
        gpus=0,
        cpus=1,
        mem="4G",
        time="30m",
        on_preempt="fail",
    )
    out = render_sbatch(
        spec,
        cluster=load_cluster("mila"),
        account="mila",
        partition="main-cpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "salvo.runtime.epilog" not in out
    assert "trap " not in out


def test_render_no_gpu_model_skips_qualifier():
    """Cluster without gpu_model emits plain --gres=gpu:N (no model qualifier)."""
    cluster = Cluster(
        schema_version=1,
        id="generic",
        display_name="Generic",
        type="slurm",
        gpu_model=None,
        accounts=[Account(name="acc", gpu_cap=4, cpu_cap=16, mem_cap_gb=64)],
        partitions=[Partition(name="gpu", max_walltime_hours=24, max_gpus_per_job=4)],
    )
    spec = JobSpec(
        name="gpu",
        cmd=["python", "train.py"],
        gpus=2,
        cpus=4,
        mem="16G",
        time="1h",
    )
    out = render_sbatch(
        spec,
        cluster=cluster,
        account="acc",
        partition="gpu",
        artifact_dir="/tmp/.salvo/runs/JOBID",
    )
    assert "#SBATCH --gres=gpu:2" in out
    assert "--gres=gpu::" not in out
    assert "h100" not in out
    assert "a100" not in out
