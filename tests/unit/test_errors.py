from salvo.errors import (
    ClusterYAMLError,
    DataNotStagedError,
    DispatchError,
    EventSchemaError,
    ManifestError,
    MaxHopsExceededError,
    NoAccountError,
    NoPartitionError,
    OomPolicyError,
    SalvoError,
    SpecValidationError,
    SubprocessError,
)


def test_hierarchy():
    assert issubclass(DataNotStagedError, SalvoError)
    assert issubclass(DispatchError, SalvoError)
    assert issubclass(NoAccountError, DispatchError)
    assert issubclass(NoPartitionError, DispatchError)
    assert issubclass(MaxHopsExceededError, SalvoError)
    assert issubclass(ClusterYAMLError, SalvoError)
    assert issubclass(SubprocessError, SalvoError)
    assert issubclass(OomPolicyError, SalvoError)
    assert issubclass(EventSchemaError, SalvoError)
    assert issubclass(ManifestError, SalvoError)
    assert issubclass(SpecValidationError, SalvoError)


def test_subprocess_error_attrs():
    e = SubprocessError("squeue failed", cmd=["squeue", "-u", "me"], returncode=1, stderr="boom")
    assert e.cmd == ["squeue", "-u", "me"]
    assert e.returncode == 1
    assert e.stderr == "boom"
    assert "squeue failed" in str(e)


def test_public_reexports_include_new_subclasses():
    """salvo.__all__ must include the new SalvoError subclasses for catchability."""
    import salvo

    for name in (
        "SalvoError",
        "SubprocessError",
        "OomPolicyError",
        "EventSchemaError",
        "ManifestError",
        "SpecValidationError",
        "DataNotStagedError",
        "DispatchError",
        "NoAccountError",
        "NoPartitionError",
        "MaxHopsExceededError",
        "ClusterYAMLError",
    ):
        assert name in salvo.__all__, name
        assert hasattr(salvo, name), name


def test_data_not_staged_message():
    e = DataNotStagedError(cluster="rorqual", missing={"mrms"}, sources={"mrms": ["mila"]})
    msg = str(e)
    assert "rorqual" in msg
    assert "mrms" in msg
    assert "mila" in msg
