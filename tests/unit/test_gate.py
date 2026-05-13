from datetime import UTC, datetime

import pytest
from salvo.errors import DataNotStagedError
from salvo.job.spec import JobSpec
from salvo.manifest.schema import DatasetLocation
from salvo.manifest.store import Manifest
from salvo.stage.gate import assert_data_available


def _m(tmp_path):
    m = Manifest.load(tmp_path / "m.toml")
    loc = DatasetLocation(path="/x", verified_at=datetime.now(UTC))
    m.record("mrms", cluster="mila", location=loc)
    return m


def test_pass(tmp_path):
    spec = JobSpec(name="t", cmd=["echo"], data_needs=["mrms"])
    assert_data_available(spec, cluster_id="mila", manifest=_m(tmp_path))


def test_missing_raises(tmp_path):
    spec = JobSpec(name="t", cmd=["echo"], data_needs=["mrms", "eastasia"])
    with pytest.raises(DataNotStagedError) as exc:
        assert_data_available(spec, cluster_id="mila", manifest=_m(tmp_path))
    assert "eastasia" in str(exc.value)


def test_optional_missing_no_raise(tmp_path):
    spec = JobSpec(name="t", cmd=["echo"], data_needs=[], data_optional=["eastasia"])
    assert_data_available(spec, cluster_id="mila", manifest=_m(tmp_path))


def test_override_allowed(tmp_path):
    spec = JobSpec(name="t", cmd=["echo"], data_needs=["mrms"])
    assert_data_available(spec, cluster_id="rorqual", manifest=_m(tmp_path), allow_missing=True)
