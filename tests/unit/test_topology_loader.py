from pathlib import Path

import pytest
from salvo.errors import ClusterYAMLError
from salvo.topology.loader import load_from_path

FIXTURES = Path(__file__).parent.parent / "fixtures" / "clusters"


def test_load_from_path(tmp_path):
    c = load_from_path(FIXTURES / "child.yaml", search_dirs=[FIXTURES])
    assert c.id == "child"
    assert c.cpus_per_gpu == 16  # inherited from parent
    assert {a.name for a in c.accounts} == {"shared-acct", "my-acct"}
    assert len(c.partitions) == 1


def test_unknown_extends_raises(tmp_path):
    f = tmp_path / "orphan.yaml"
    f.write_text("schema_version: 1\nid: o\ndisplay_name: O\ntype: local\nextends: missing\n")
    with pytest.raises(ClusterYAMLError, match="missing"):
        load_from_path(f, search_dirs=[tmp_path])


def test_invalid_schema_raises(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("schema_version: 1\nid: b\ndisplay_name: B\ntype: bogus\n")
    with pytest.raises(ClusterYAMLError):
        load_from_path(f, search_dirs=[tmp_path])
