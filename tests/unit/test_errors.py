from salvo.errors import (
    ClusterYAMLError,
    DataNotStagedError,
    DispatchError,
    MaxHopsExceededError,
    NoAccountError,
    NoPartitionError,
    SalvoError,
)


def test_hierarchy():
    assert issubclass(DataNotStagedError, SalvoError)
    assert issubclass(DispatchError, SalvoError)
    assert issubclass(NoAccountError, DispatchError)
    assert issubclass(NoPartitionError, DispatchError)
    assert issubclass(MaxHopsExceededError, SalvoError)
    assert issubclass(ClusterYAMLError, SalvoError)


def test_data_not_staged_message():
    e = DataNotStagedError(cluster="rorqual", missing={"mrms"}, sources={"mrms": ["mila"]})
    msg = str(e)
    assert "rorqual" in msg
    assert "mrms" in msg
    assert "mila" in msg
