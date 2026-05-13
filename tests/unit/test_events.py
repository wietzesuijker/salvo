import json

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
