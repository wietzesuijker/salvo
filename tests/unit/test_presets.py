import pytest
from salvo.topology.loader import load_cluster

PRESETS_TO_VERIFY = ["mila", "rorqual", "narval", "beluga", "cedar"]


@pytest.mark.parametrize("cid", PRESETS_TO_VERIFY)
def test_preset_loads(cid):
    c = load_cluster(cid)
    assert c.id == cid
    assert c.accounts, f"{cid} has no accounts"
    assert c.partitions, f"{cid} has no partitions"
    assert c.schema_version == 1


def test_mila_has_main_cpu_64g_ceiling():
    c = load_cluster("mila")
    p = next(p for p in c.partitions if p.name == "main-cpu")
    assert p.max_mem_gb == 64


def test_mila_has_ssh_alias():
    c = load_cluster("mila")
    assert c.login is not None
    assert c.login.ssh_alias == "mila"


def test_rorqual_ctb_drolnick_is_gpu_only():
    c = load_cluster("rorqual")
    a = next(a for a in c.accounts if a.name == "ctb-drolnick")
    assert a.gpu_only is True


def test_rorqual_forbids_drolnick_for_cpu_only():
    c = load_cluster("rorqual")
    rule = next(r for r in c.dispatch_rules if r.when.get("gpus") == 0)
    assert "ctb-drolnick" in rule.forbid


def test_drac_login_has_apptainer_env():
    for cid in ["rorqual", "narval", "beluga", "cedar"]:
        c = load_cluster(cid)
        assert c.login is not None
        assert c.login.apptainer_env.get("GOMAXPROCS") == "1"
        assert c.login.apptainer_env.get("GOMEMLIMIT") == "2GiB"


def test_rorqual_bygpu_uses_per_gpu_mem():
    c = load_cluster("rorqual")
    for pname in ("gpubase_bygpu_b1", "gpubase_bygpu_b3"):
        p = next(p for p in c.partitions if p.name == pname)
        assert p.prefer_mem_kind == "per-gpu", f"{pname} must use per-gpu mem"


def test_mila_long_requires_qos():
    c = load_cluster("mila")
    p = next(p for p in c.partitions if p.name == "long")
    assert p.requires_qos == "long"


def test_mila_long_cpu_requires_qos():
    c = load_cluster("mila")
    p = next(p for p in c.partitions if p.name == "long-cpu")
    assert p.requires_qos == "long"
