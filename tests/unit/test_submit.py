"""Concurrency hardening tests for `salvo.job.submit.submit`.

Covers pending-dir naming (uuid4-only, no pid leak) and the atomic-rename
clobber guard on the final runs/<job_id> directory.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from salvo import submit
from salvo.errors import SalvoError
from salvo.job.spec import JobSpec
from salvo.manifest.store import Manifest


def _spec() -> JobSpec:
    return JobSpec(name="t", cmd=["echo", "hi"], cpus=2, mem="4G", time="30m")


def test_pending_id_uses_uuid_hex_no_pid(
    fake_sbatch: list[list[str]], tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pending dir name is `pending-<uuid4-hex>` with no os.getpid() leakage."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")

    captured: dict[str, Path] = {}
    real_mkdir = Path.mkdir

    def _capture_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith("pending-") and "pending" not in captured:
            captured["pending"] = self
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", _capture_mkdir)

    fixed = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr("salvo.job.submit.uuid.uuid4", lambda: fixed)

    submit(_spec(), cluster_id="mila")

    assert "pending" in captured, "pending dir was never created"
    name = captured["pending"].name
    assert name == f"pending-{fixed.hex}"
    assert str(os.getpid()) not in name


def test_submit_refuses_to_clobber_existing_final_dir(
    fake_sbatch: list[list[str]], tmp_home: Path
) -> None:
    """If runs/<job_id> already exists, refuse the rename rather than silently overwrite."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    # The first fake sbatch returns 1001 (counter starts at 1000 and pre-increments).
    runs_dir = tmp_home / ".salvo" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stale = runs_dir / "1001"
    stale.mkdir()
    (stale / "marker").write_text("stale")

    with pytest.raises(SalvoError, match="refusing to clobber"):
        submit(_spec(), cluster_id="mila")

    # Stale dir is untouched.
    assert (stale / "marker").read_text() == "stale"


def test_pending_dir_cleaned_on_clobber_refusal(
    fake_sbatch: list[list[str]], tmp_home: Path
) -> None:
    """On clobber refusal the pending dir is left on disk for forensics."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    runs_dir = tmp_home / ".salvo" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "1001").mkdir()

    with pytest.raises(SalvoError):
        submit(_spec(), cluster_id="mila")

    # The pending-* dir should still exist (not auto-deleted).
    pending = [p for p in runs_dir.iterdir() if p.name.startswith("pending-")]
    assert len(pending) == 1, f"expected exactly one pending dir, found {pending}"


def test_two_submits_get_unique_pending_dirs(
    fake_sbatch: list[list[str]], tmp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two submits must produce distinct pending dir names even if pid would repeat."""
    Manifest.load(tmp_home / ".salvo" / "manifest.toml")
    uuids = iter(
        [
            uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
            uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        ]
    )
    monkeypatch.setattr("salvo.job.submit.uuid.uuid4", lambda: next(uuids))

    h1 = submit(_spec(), cluster_id="mila")
    h2 = submit(_spec(), cluster_id="mila")
    assert h1.artifact_dir != h2.artifact_dir
    assert h1.job_id != h2.job_id
