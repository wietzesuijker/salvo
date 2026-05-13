from datetime import UTC, datetime

from salvo.manifest.schema import DatasetLocation
from salvo.manifest.store import Manifest


def test_create_empty(tmp_path):
    m = Manifest.load(tmp_path / "manifest.toml")
    assert m.datasets == {}


def test_record_and_query(tmp_path):
    p = tmp_path / "m.toml"
    m = Manifest.load(p)
    m.record(
        "mrms",
        cluster="mila",
        location=DatasetLocation(
            path="/network/scratch/u/me/mrms",
            verified_at=datetime.now(UTC),
            size_gb=412,
        ),
    )
    m.save()
    m2 = Manifest.load(p)
    assert "mrms" in m2.datasets
    assert m2.datasets["mrms"].locations["mila"].size_gb == 412


def test_locations_on(tmp_path):
    m = Manifest.load(tmp_path / "m.toml")
    m.record(
        "a", cluster="mila", location=DatasetLocation(path="/x", verified_at=datetime.now(UTC))
    )
    m.record(
        "a", cluster="rorqual", location=DatasetLocation(path="/y", verified_at=datetime.now(UTC))
    )
    assert set(m.locations_on("mila")) == {"a"}
    assert set(m.which_clusters_have({"a"})["a"]) == {"mila", "rorqual"}


def test_atomic_write_via_rename(tmp_path, monkeypatch):
    p = tmp_path / "m.toml"
    m = Manifest.load(p)
    m.record(
        "a", cluster="mila", location=DatasetLocation(path="/x", verified_at=datetime.now(UTC))
    )
    m.save()
    # tmp file should be removed; target exists
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp.*"))
