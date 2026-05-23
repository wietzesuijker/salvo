"""Tests for hop+parent persistence on the submit path (`salvo.job.submit.submit`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from salvo import submit
from salvo.job.spec import JobSpec
from salvo.manifest.store import Manifest


def _spec() -> JobSpec:
    return JobSpec(name="t", cmd=["echo", "hi"], cpus=2, mem="4G", time="30m")


def test_submit_writes_hop_file_when_hop_set(fake_sbatch: list[list[str]], tmp_home: Path) -> None:
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    h = submit(_spec(), cluster_id="mila", hop="2/5")
    hop_file = h.artifact_dir / "hop.json"
    assert hop_file.exists()
    data = json.loads(hop_file.read_text())
    assert data["hop"] == "2/5"


def test_submit_writes_parent_json_when_parent_passed(
    fake_sbatch: list[list[str]], tmp_home: Path
) -> None:
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    h1 = submit(_spec(), cluster_id="mila")
    parent_job_id = h1.job_id

    h2 = submit(_spec(), cluster_id="mila", parent_artifact_dir=h1.artifact_dir)
    parent_file = h2.artifact_dir / "parent.json"
    assert parent_file.exists()
    data = json.loads(parent_file.read_text())
    assert data["parent_artifact_dir"] == str(h1.artifact_dir)
    assert data["parent_job_id"] == parent_job_id


def test_submit_no_parent_no_hop_default(fake_sbatch: list[list[str]], tmp_home: Path) -> None:
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    h = submit(_spec(), cluster_id="mila")
    assert not (h.artifact_dir / "hop.json").exists()
    assert not (h.artifact_dir / "parent.json").exists()


def test_submit_parent_json_resolves_parent_job_id_from_events(
    fake_sbatch: list[list[str]], tmp_home: Path
) -> None:
    """parent_job_id comes from the submit.success event line in parent's events.jsonl."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    h1 = submit(_spec(), cluster_id="mila")
    # h1.job_id should appear in parent.json after a child submission.
    h2 = submit(_spec(), cluster_id="mila", parent_artifact_dir=h1.artifact_dir)
    data = json.loads((h2.artifact_dir / "parent.json").read_text())
    assert data["parent_job_id"] == h1.job_id


def test_submit_hop_and_parent_persist_together(
    fake_sbatch: list[list[str]], tmp_home: Path
) -> None:
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    h1 = submit(_spec(), cluster_id="mila")
    h2 = submit(
        _spec(),
        cluster_id="mila",
        hop="3/5",
        parent_artifact_dir=h1.artifact_dir,
    )
    assert (h2.artifact_dir / "hop.json").exists()
    assert (h2.artifact_dir / "parent.json").exists()


def test_submit_hop_kwarg_is_keyword_only(fake_sbatch: list[list[str]], tmp_home: Path) -> None:
    """Positional-only signature guards against accidental misuse."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    spec = _spec()
    with pytest.raises(TypeError):
        submit(spec, "mila", "2/5")  # type: ignore[misc]
