"""Subprocess error wrapping: FileNotFoundError, CalledProcessError, TimeoutExpired
all surface as SubprocessError (a SalvoError) with cmd/returncode/stderr attributes.

TDD discipline: these tests are written first; the helper they exercise must
catch the three subprocess exceptions and re-raise as SubprocessError so callers
only ever see SalvoError subclasses.
"""

from __future__ import annotations

import subprocess

import pytest
from salvo.errors import SalvoError, SubprocessError, _run_subprocess


def test_file_not_found_wrapped(monkeypatch):
    def _raise(*a, **kw):
        raise FileNotFoundError(2, "No such file or directory", "squeue")

    monkeypatch.setattr("subprocess.run", _raise)
    with pytest.raises(SubprocessError) as exc:
        _run_subprocess(["squeue", "-u", "me"])
    assert isinstance(exc.value, SalvoError)
    assert exc.value.cmd == ["squeue", "-u", "me"]
    assert exc.value.returncode is None
    assert "squeue" in str(exc.value)
    assert "PATH" in str(exc.value)


def test_called_process_error_wrapped(monkeypatch):
    def _raise(*a, **kw):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=["sbatch", "x.sh"], stderr="permission denied"
        )

    monkeypatch.setattr("subprocess.run", _raise)
    with pytest.raises(SubprocessError) as exc:
        _run_subprocess(["sbatch", "x.sh"])
    assert isinstance(exc.value, SalvoError)
    assert exc.value.cmd == ["sbatch", "x.sh"]
    assert exc.value.returncode == 1
    assert "permission denied" in exc.value.stderr
    assert "sbatch" in str(exc.value)


def test_timeout_wrapped(monkeypatch):
    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=["squeue"], timeout=10)

    monkeypatch.setattr("subprocess.run", _raise)
    with pytest.raises(SubprocessError) as exc:
        _run_subprocess(["squeue"], timeout=10)
    assert isinstance(exc.value, SalvoError)
    assert exc.value.cmd == ["squeue"]
    assert exc.value.returncode is None
    assert "timed out" in str(exc.value).lower()
    assert "10" in str(exc.value)


def test_returns_completed_process(monkeypatch):
    """Happy path: helper passes through the CompletedProcess unchanged."""

    def _ok(cmd, *a, **kw):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("subprocess.run", _ok)
    proc = _run_subprocess(["squeue"])
    assert proc.returncode == 0
    assert proc.stdout == "ok\n"


def test_caps_run_squeue_wraps_filenotfound(monkeypatch):
    """Smoke: caps._run_squeue must surface SubprocessError, not FileNotFoundError."""
    from salvo.dispatch.caps import _run_squeue

    def _raise(*a, **kw):
        raise FileNotFoundError(2, "No such file", "squeue")

    monkeypatch.setattr("subprocess.run", _raise)
    with pytest.raises(SubprocessError):
        _run_squeue("me")


def test_handle_status_wraps_filenotfound(monkeypatch, tmp_path):
    """JobHandle.status() must surface SubprocessError, not FileNotFoundError."""
    from salvo.job.handle import JobHandle
    from salvo.job.spec import JobSpec

    def _raise(*a, **kw):
        raise FileNotFoundError(2, "No such file", "sacct")

    monkeypatch.setattr("subprocess.run", _raise)
    spec = JobSpec(name="t", cmd=["echo"], mem="4G", time="30m")
    h = JobHandle(job_id="123", cluster="mila", spec=spec, artifact_dir=tmp_path)
    with pytest.raises(SubprocessError):
        h.status()


def test_handle_cancel_wraps_filenotfound(monkeypatch, tmp_path):
    """JobHandle.cancel() must surface SubprocessError, not FileNotFoundError."""
    from salvo.job.handle import JobHandle
    from salvo.job.spec import JobSpec

    def _raise(*a, **kw):
        raise FileNotFoundError(2, "No such file", "scancel")

    monkeypatch.setattr("subprocess.run", _raise)
    spec = JobSpec(name="t", cmd=["echo"], mem="4G", time="30m")
    h = JobHandle(job_id="123", cluster="mila", spec=spec, artifact_dir=tmp_path)
    with pytest.raises(SubprocessError):
        h.cancel()


def test_submit_sbatch_filenotfound_wrapped(monkeypatch, tmp_path):
    """submit() shelling out to sbatch must surface SubprocessError on missing binary."""
    from salvo.job.spec import JobSpec
    from salvo.job.submit import submit

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setenv("SALVO_CLUSTER", "mila")

    real_run = subprocess.run

    def _run(cmd, *a, **kw):
        if isinstance(cmd, list | tuple) and cmd and cmd[0] == "sbatch":
            raise FileNotFoundError(2, "No such file", "sbatch")
        if isinstance(cmd, list | tuple) and cmd and cmd[0] == "squeue":
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr("subprocess.run", _run)
    spec = JobSpec(name="t", cmd=["echo", "hi"], cpus=2, mem="4G", time="30m")
    with pytest.raises(SalvoError):
        submit(spec)
