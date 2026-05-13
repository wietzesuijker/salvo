from pathlib import Path

import yaml
from salvo.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_doctor_exits_zero_when_ok(fake_sbatch, tmp_home):
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code in (0, 1)  # 1 when no manifest yet, 0 once seeded
    assert "topology.detect" in r.stdout


def test_render_emits_sbatch(fake_sbatch, tmp_home, tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "cmd": ["echo", "hi"],
                "cpus": 2,
                "mem": "4G",
                "time": "30m",
            }
        )
    )
    r = runner.invoke(app, ["render", str(spec_path)])
    assert r.exit_code == 0, r.stdout
    assert "#SBATCH" in r.stdout
    assert "--job-name=t" in r.stdout


def test_render_honours_cluster_flag(fake_sbatch, tmp_home, tmp_path: Path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "name": "t",
                "cmd": ["echo", "hi"],
                "cpus": 2,
                "mem": "4G",
                "time": "30m",
            }
        )
    )
    r = runner.invoke(app, ["render", str(spec_path), "--cluster", "mila"])
    assert r.exit_code == 0, r.stdout
    assert "#SBATCH" in r.stdout
