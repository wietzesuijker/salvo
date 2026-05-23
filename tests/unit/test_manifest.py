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


def test_load_corrupt_toml_raises_manifest_error(tmp_path):
    """Manifest.load must surface ManifestError (SalvoError) on bad TOML, not tomllib error."""
    import pytest
    from salvo.errors import ManifestError, SalvoError

    p = tmp_path / "bad.toml"
    p.write_text("this is not = valid toml [[[")
    with pytest.raises(ManifestError) as exc:
        Manifest.load(p)
    assert isinstance(exc.value, SalvoError)
    assert str(p) in str(exc.value) or "bad.toml" in str(exc.value)


def test_load_malformed_dataset_section_raises_manifest_error(tmp_path):
    """Schema-shaped errors (TypeError/KeyError from **loc) also surface as ManifestError."""
    import pytest
    from salvo.errors import ManifestError

    p = tmp_path / "shape.toml"
    # locations entry missing required 'path' field → TypeError from DatasetLocation(**loc)
    p.write_text(
        'schema_version = 1\n\n[datasets.foo.locations.mila]\nverified_at = "not-a-date"\n'
    )
    with pytest.raises(ManifestError):
        Manifest.load(p)


def test_dataset_is_frozen():
    """Dataset must be frozen + extra='forbid' to prevent accidental schema drift."""
    import pytest
    from pydantic import ValidationError
    from salvo.manifest.schema import Dataset

    ds = Dataset(description="x")
    with pytest.raises(ValidationError):
        ds.description = "y"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Dataset(description="x", bogus_field=1)  # type: ignore[call-arg]


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
