"""JobHandle.status() must parse sacct output including the .batch step row
where MaxRSS is reported.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from salvo.job.handle import JobHandle
from salvo.job.spec import JobSpec


def _spec() -> JobSpec:
    return JobSpec(name="t", cmd=["echo"], mem="4G", time="30m")


def _handle(tmp_path: Path) -> JobHandle:
    return JobHandle(job_id="123", cluster="mila", spec=_spec(), artifact_dir=tmp_path)


def _sacct(stdout: str):
    def _run(cmd, *a, **kw):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    return _run


def test_status_parses_maxrss_from_batch_step(monkeypatch, tmp_path):
    out = (
        "123|COMPLETED|0:0|00:05:00|\n"
        "123.batch|COMPLETED|0:0|00:05:00|512000K\n"
        "123.extern|COMPLETED|0:0|00:05:00|\n"
    )
    monkeypatch.setattr("subprocess.run", _sacct(out))
    s = _handle(tmp_path).status()
    assert s.job_id == "123"
    assert s.state == "COMPLETED"
    assert s.max_rss_kb == 512000


def test_status_handles_no_batch_row(monkeypatch, tmp_path):
    out = "123|RUNNING|0:0|00:01:00|\n"
    monkeypatch.setattr("subprocess.run", _sacct(out))
    s = _handle(tmp_path).status()
    assert s.state == "RUNNING"
    assert s.max_rss_kb is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1024K", 1024),
        ("4096M", 4096 * 1024),
        ("2G", 2 * 1024 * 1024),
        ("", None),
    ],
)
def test_status_handles_max_rss_units(monkeypatch, tmp_path, raw, expected):
    out = f"123|COMPLETED|0:0|00:05:00|\n123.batch|COMPLETED|0:0|00:05:00|{raw}\n"
    monkeypatch.setattr("subprocess.run", _sacct(out))
    s = _handle(tmp_path).status()
    assert s.max_rss_kb == expected


def test_status_drops_x_flag(monkeypatch, tmp_path):
    """sacct invocation must NOT pass -X (would hide .batch step row)."""
    captured: dict[str, list[str]] = {}

    def _run(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", _run)
    _handle(tmp_path).status()
    assert "-X" not in captured["cmd"]
