from unittest.mock import patch

import pytest
from salvo.dispatch.caps import CapsTracker, _parse_mem

SQUEUE_OUT = """\
12345|rrg-bengioy-ad|gpubase_bygpu_b3|gpu:2|16|32768
12346|rrg-bengioy-ad|cpubase_bycore_b1||8|16384
"""


def test_parses_squeue():
    with patch("salvo.dispatch.caps._run_squeue", return_value=SQUEUE_OUT):
        t = CapsTracker(cluster_id="rorqual", user="me", ttl_sec=60)
        u = t.snapshot()
        assert u.gpus_in_use["rrg-bengioy-ad"] == 2
        assert u.cpus_in_use["rrg-bengioy-ad"] == 24
        assert u.mem_in_use_mb["rrg-bengioy-ad"] == 49152


def test_counter_increments_atomically():
    with patch("salvo.dispatch.caps._run_squeue", return_value=""):
        t = CapsTracker(cluster_id="x", user="me", ttl_sec=60)
        t.account_after_submit("rrg-bengioy-ad", gpus=1, cpus=8, mem_mb=16384)
        u = t.snapshot()
        assert u.gpus_in_use["rrg-bengioy-ad"] == 1
        assert u.cpus_in_use["rrg-bengioy-ad"] == 8


def test_ttl_refresh(monkeypatch):
    calls = []

    def fake(*a, **kw):
        calls.append(1)
        return ""

    monkeypatch.setattr("salvo.dispatch.caps._run_squeue", fake)
    monkeypatch.setattr("time.monotonic", lambda: 0)
    t = CapsTracker(cluster_id="x", user="me", ttl_sec=60)
    t.snapshot()
    monkeypatch.setattr("time.monotonic", lambda: 30)
    t.snapshot()
    assert len(calls) == 1
    monkeypatch.setattr("time.monotonic", lambda: 61)
    t.snapshot()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "s,mb",
    [
        ("4G", 4096),
        ("4000M", 4000),
        ("2048K", 2),
        ("1T", 1048576),
        ("", 0),
        ("garbage", 0),
        ("4GiB", 4096),
    ],
)
def test_parse_mem(s, mb):
    assert _parse_mem(s) == mb
