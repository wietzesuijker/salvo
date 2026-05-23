"""Tests for `salvo.runtime.epilog`: SIGUSR1 trap handler that resubmits preempted jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from salvo.errors import SalvoError
from salvo.job.spec import JobSpec
from salvo.runtime import epilog


def _write_parent_artifacts(
    parent_dir: Path, *, hop: str = "2/5", job_id: str = "11111"
) -> JobSpec:
    parent_dir.mkdir(parents=True, exist_ok=True)
    spec = JobSpec(
        name="t",
        cmd=["python", "-c", "x"],
        cpus=2,
        mem="4G",
        time="1h",
        on_preempt="resubmit",
        max_hops=5,
    )
    (parent_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
    # cluster.json minimally records the cluster_id used at submit time.
    (parent_dir / "cluster.json").write_text(json.dumps({"id": "mila"}))
    # events.jsonl carries the parent job_id from the submit.success event.
    (parent_dir / "events.jsonl").write_text(
        json.dumps({"event": "submit.success", "job_id": job_id, "ts": "2026-05-22T00:00:00Z"})
        + "\n"
    )
    return spec


def test_epilog_max_hops_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="5/5")
    monkeypatch.setenv("SALVO_HOP", "5/5")

    submit_mock = MagicMock()
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 0
    submit_mock.assert_not_called()

    events = (parent_dir / "events.jsonl").read_text().splitlines()
    last = json.loads(events[-1])
    assert last["event"] == "hop.max_exceeded"


def test_epilog_resubmits_on_preempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="2/5", job_id="11111")
    monkeypatch.setenv("SALVO_HOP", "2/5")

    child_dir = tmp_path / "child"
    child_dir.mkdir()
    fake_handle = MagicMock()
    fake_handle.job_id = "22222"
    fake_handle.artifact_dir = child_dir

    submit_mock = MagicMock(return_value=fake_handle)
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 0

    # submit() was called with hop="3/5", parent_artifact_dir=parent, cluster_id from json.
    submit_mock.assert_called_once()
    kwargs = submit_mock.call_args.kwargs
    assert kwargs["hop"] == "3/5"
    assert kwargs["parent_artifact_dir"] == parent_dir
    assert kwargs["cluster_id"] == "mila"

    # Hop record landed in parent's hops.jsonl.
    hops_file = parent_dir / "hops.jsonl"
    assert hops_file.exists()
    line = hops_file.read_text().strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["parent_job_id"] == "11111"
    assert rec["child_job_id"] == "22222"
    assert rec["cause"] == "preempt"
    assert rec["hop_index"] == 3

    # An event marks the resubmit.
    events = (parent_dir / "events.jsonl").read_text().splitlines()
    last_events = [json.loads(e) for e in events]
    assert any(e["event"] == "hop.preempt" for e in last_events)


def test_epilog_missing_spec_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    # No spec.json written.
    (parent_dir / "events.jsonl").write_text("")
    monkeypatch.setenv("SALVO_HOP", "1/5")

    submit_mock = MagicMock()
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 1
    submit_mock.assert_not_called()


def test_epilog_missing_artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "does-not-exist"
    monkeypatch.setenv("SALVO_HOP", "0/5")

    submit_mock = MagicMock()
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 1
    submit_mock.assert_not_called()


def test_epilog_handles_submit_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="0/5")
    monkeypatch.setenv("SALVO_HOP", "0/5")

    def _boom(*a: Any, **kw: Any) -> None:
        raise SalvoError("sbatch died")

    monkeypatch.setattr("salvo.runtime.epilog._submit", _boom)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 1


def test_epilog_main_cli_invokes_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="5/5")
    monkeypatch.setenv("SALVO_HOP", "5/5")
    monkeypatch.setattr("salvo.runtime.epilog._submit", MagicMock())

    with pytest.raises(SystemExit) as exc:
        epilog.main(["--artifact-dir", str(parent_dir), "--reason", "preempt"])
    assert exc.value.code == 0


def test_epilog_missing_cluster_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="1/5")
    (parent_dir / "cluster.json").unlink()
    monkeypatch.setenv("SALVO_HOP", "1/5")

    submit_mock = MagicMock()
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 1
    submit_mock.assert_not_called()
    events = (parent_dir / "events.jsonl").read_text().splitlines()
    assert any("cluster.json missing" in json.loads(e).get("error", "") for e in events)


def test_epilog_cluster_json_missing_id_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="1/5")
    (parent_dir / "cluster.json").write_text(json.dumps({}))
    monkeypatch.setenv("SALVO_HOP", "1/5")

    submit_mock = MagicMock()
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 1
    submit_mock.assert_not_called()


def test_epilog_unknown_parent_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If events.jsonl lacks submit.success, parent_job_id falls back to 'unknown'."""
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    spec = JobSpec(
        name="t",
        cmd=["python", "-c", "x"],
        cpus=2,
        mem="4G",
        time="1h",
        on_preempt="resubmit",
        max_hops=5,
    )
    (parent_dir / "spec.json").write_text(spec.model_dump_json(indent=2))
    (parent_dir / "cluster.json").write_text(json.dumps({"id": "mila"}))
    # events.jsonl present but no submit.success line.
    (parent_dir / "events.jsonl").write_text(
        json.dumps({"event": "submit.attempt", "ts": "2026-05-22T00:00:00Z"})
        + "\n"
        # JSON-decode failure line (defensive coverage):
        + "not-json\n"
    )
    monkeypatch.setenv("SALVO_HOP", "0/5")

    child_dir = tmp_path / "child3"
    child_dir.mkdir()
    fake_handle = MagicMock()
    fake_handle.job_id = "55555"
    fake_handle.artifact_dir = child_dir

    submit_mock = MagicMock(return_value=fake_handle)
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 0
    rec = json.loads((parent_dir / "hops.jsonl").read_text().strip())
    assert rec["parent_job_id"] == "unknown"


def test_epilog_no_hop_env_defaults_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SALVO_HOP unset → treat as `0/{max}` and resubmit."""
    parent_dir = tmp_path / "parent"
    _write_parent_artifacts(parent_dir, hop="0/5", job_id="33333")
    monkeypatch.delenv("SALVO_HOP", raising=False)

    child_dir = tmp_path / "child2"
    child_dir.mkdir()
    fake_handle = MagicMock()
    fake_handle.job_id = "44444"
    fake_handle.artifact_dir = child_dir

    submit_mock = MagicMock(return_value=fake_handle)
    monkeypatch.setattr("salvo.runtime.epilog._submit", submit_mock)

    rc = epilog.run(artifact_dir=parent_dir, reason="preempt")
    assert rc == 0
    kwargs = submit_mock.call_args.kwargs
    assert kwargs["hop"] == "1/5"
