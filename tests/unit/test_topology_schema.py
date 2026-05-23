import pytest
from pydantic import ValidationError
from salvo.topology.schema import (
    Account,
    Cluster,
    DispatchRule,
    LoginConstraints,
    Partition,
)


def test_minimal_cluster():
    c = Cluster(
        schema_version=1,
        id="x",
        display_name="X",
        type="local",
        accounts=[Account(name="me", gpu_cap=1, cpu_cap=8, mem_cap_gb=32)],
        partitions=[Partition(name="main", max_walltime_hours=24, max_gpus_per_job=4)],
    )
    assert c.id == "x"


def test_dispatch_rule_when_clause():
    r = DispatchRule(when={"gpus": ">0"}, prefer=["a"], reason="r")
    assert r.matches(gpus=1, cpus=1, mem_mb=1024) is True
    assert r.matches(gpus=0, cpus=1, mem_mb=1024) is False

    r2 = DispatchRule(when={"gpus": 0}, prefer=["a"], reason="r")
    assert r2.matches(gpus=0) is True
    assert r2.matches(gpus=1) is False


def test_account_gpu_only_flag():
    a = Account(name="ctb-drolnick", gpu_cap=16, cpu_cap=0, mem_cap_gb=0, gpu_only=True)
    assert a.gpu_only is True


def test_login_constraints_apptainer_env():
    lc = LoginConstraints(
        ssh_alias="rorqual",
        apptainer_env={"GOMAXPROCS": "1", "GOMEMLIMIT": "2GiB"},
        reason="pids.max=512",
    )
    assert lc.apptainer_env["GOMAXPROCS"] == "1"


def test_schema_version_required():
    with pytest.raises(ValidationError):
        Cluster(id="x", display_name="X", type="local", accounts=[], partitions=[])  # missing


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Cluster(
            schema_version=1,
            id="x",
            display_name="X",
            type="local",
            accounts=[],
            partitions=[],
            bogus=True,
        )


def test_partition_defaults_prefer_total():
    p = Partition(name="main", max_walltime_hours=3, max_gpus_per_job=4)
    assert p.prefer_mem_kind == "total"
    assert p.requires_qos is None


def test_partition_rejects_invalid_prefer_mem_kind():
    with pytest.raises(ValidationError):
        Partition(
            name="bad",
            max_walltime_hours=3,
            max_gpus_per_job=4,
            prefer_mem_kind="per-cpu",
        )
