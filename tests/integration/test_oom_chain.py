"""Integration: simulate CPU OOM, verify resubmit chain bumps mem in the new sbatch."""

from salvo import JobSpec, submit
from salvo.job.oom import OomContext, apply_oom_policy


def test_resubmit_with_bumped_mem(fake_sbatch, tmp_home):
    spec = JobSpec(
        name="oom_t",
        cmd=["python", "-c", "x"],
        cpus=2,
        mem="16G",
        time="1h",
        on_oom=["bump_mem(2x, max=128G)", "fail"],
    )
    h1 = submit(spec, cluster_id="mila")
    new_spec, action = apply_oom_policy(h1.spec, OomContext(class_="cpu", max_rss_mb=15500))
    assert new_spec is not None
    assert action == "bump_mem"
    assert new_spec.mem_mb() >= 32 * 1024

    h2 = submit(new_spec, cluster_id="mila")
    assert h2.job_id != h1.job_id
    sbatch2 = (h2.artifact_dir / "sbatch.sh").read_text()
    assert f"--mem={new_spec.mem_mb()}M" in sbatch2
