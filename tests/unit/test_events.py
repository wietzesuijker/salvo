import fcntl
import json
import multiprocessing as mp
import os

from salvo.obs.events import EventEmitter


def test_emits_jsonl_line(tmp_path):
    em = EventEmitter(tmp_path / "events.jsonl")
    em.emit(
        "submit.success",
        job_id="123",
        cluster="mila",
        account="mila",
        partition="main",
        gpus=1,
        mem_mb=16384,
        time_min=60,
        salvo_version="0.0.0",
    )
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["event"] == "submit.success"
    assert obj["job_id"] == "123"
    assert "ts" in obj


def test_appends_not_overwrites(tmp_path):
    em = EventEmitter(tmp_path / "events.jsonl")
    em.emit("submit.attempt", job_id="123")
    em.emit("submit.success", job_id="123")
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_unknown_event_rejected(tmp_path):
    em = EventEmitter(tmp_path / "events.jsonl")
    import pytest

    with pytest.raises(ValueError):
        em.emit("bogus.event", job_id="x")


def test_unknown_event_raises_event_schema_error(tmp_path):
    """Unknown event must surface as EventSchemaError (SalvoError), not bare ValueError."""
    import pytest
    from salvo.errors import EventSchemaError, SalvoError

    em = EventEmitter(tmp_path / "events.jsonl")
    with pytest.raises(EventSchemaError) as exc:
        em.emit("bogus.event", job_id="x")
    assert isinstance(exc.value, SalvoError)
    # back-compat: still IS-A ValueError so existing callers keep working
    assert isinstance(exc.value, ValueError)


def test_events_flock_uses_exclusive_lock(tmp_path, monkeypatch):
    """emit() must acquire LOCK_EX and release with LOCK_UN."""
    calls: list[tuple[int, int]] = []
    real_flock = fcntl.flock

    def _record(fd, op):
        calls.append((fd, op))
        return real_flock(fd, op)

    monkeypatch.setattr("salvo.obs.events.fcntl.flock", _record)

    em = EventEmitter(tmp_path / "events.jsonl")
    em.emit("submit.attempt", job_id="123")

    ops = [op for _fd, op in calls]
    assert fcntl.LOCK_EX in ops
    assert fcntl.LOCK_UN in ops
    # LOCK_EX must come before LOCK_UN.
    assert ops.index(fcntl.LOCK_EX) < ops.index(fcntl.LOCK_UN)
    # Same fd for both calls (locking the file we just wrote to).
    fds = {fd for fd, _ in calls}
    assert len(fds) == 1


def test_events_fsync_called(tmp_path, monkeypatch):
    """emit() must fsync the fd after writing so audit log is durable."""
    calls: list[int] = []
    real_fsync = os.fsync

    def _record(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr("salvo.obs.events.os.fsync", _record)

    em = EventEmitter(tmp_path / "events.jsonl")
    em.emit("submit.attempt", job_id="123")

    assert len(calls) == 1
    assert isinstance(calls[0], int)


def _mp_emit_worker(args):
    """Top-level worker so multiprocessing can serialize it for spawn or fork."""
    path, n, tag = args
    em = EventEmitter(path)
    for i in range(n):
        em.emit("submit.attempt", job_id=f"{tag}-{i}")


def test_events_concurrent_writes_serialize(tmp_path):
    """4 processes x 25 events each = 100 valid JSON lines, no torn writes.

    Multiprocessing chosen over threading: the GIL would mask the cross-process
    race the flock is designed to prevent.
    """
    path = tmp_path / "events.jsonl"
    args = [(path, 25, f"w{i}") for i in range(4)]
    with mp.Pool(4) as pool:
        pool.map(_mp_emit_worker, args)

    lines = path.read_text().splitlines()
    assert len(lines) == 100
    for line in lines:
        obj = json.loads(line)  # raises if any line is garbled / interleaved
        assert obj["event"] == "submit.attempt"
        assert "job_id" in obj
