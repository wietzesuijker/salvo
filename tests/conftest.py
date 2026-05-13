"""Top-level fixtures: temp HOME, fake sbatch/squeue/sacct."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SALVO_CLUSTER", "mila")
    return home


@pytest.fixture
def fake_sbatch(monkeypatch):
    """Replace subprocess.run for sbatch invocations; returns a deterministic job_id."""
    counter = {"i": 1000}
    calls: list[list[str]] = []

    def _run(cmd, *args, **kwargs):
        calls.append(list(cmd) if isinstance(cmd, list | tuple) else [cmd])
        result = MagicMock()
        if isinstance(cmd, list | tuple) and cmd[0] == "sbatch":
            result.returncode = 0
            counter["i"] += 1
            result.stdout = f"Submitted batch job {counter['i']}\n"
            result.stderr = ""
        elif isinstance(cmd, list | tuple) and cmd[0] == "squeue":
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
        elif isinstance(cmd, list | tuple) and cmd[0] == "sacct":
            result.returncode = 0
            result.stdout = "JobID|State|ExitCode\n"
            result.stderr = ""
        else:
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
        return result

    monkeypatch.setattr("subprocess.run", _run)
    return calls
