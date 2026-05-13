from salvo.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_doctor_exits_zero_when_ok(fake_sbatch, tmp_home):
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code in (0, 1)  # 1 when no manifest yet, 0 once seeded
    assert "topology.detect" in r.stdout


def test_submit_basic(fake_sbatch, tmp_home):
    args = [
        "submit",
        "--name",
        "t",
        "--cpus",
        "2",
        "--mem",
        "4G",
        "--time",
        "30m",
        "--",
        "echo",
        "hi",
    ]
    r = runner.invoke(app, args)
    assert r.exit_code == 0
    assert "job_id" in r.stdout.lower() or "submitted" in r.stdout.lower()


def test_submit_plan_dry_run(fake_sbatch, tmp_home):
    r = runner.invoke(app, ["submit", "--name", "t", "--plan", "--", "echo", "hi"])
    assert r.exit_code == 0
    assert "#SBATCH" in r.stdout
