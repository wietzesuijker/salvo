import pytest
from salvo.topology.detect import detect_cluster


def test_explicit_env_wins(monkeypatch):
    monkeypatch.setenv("SALVO_CLUSTER", "narval")
    monkeypatch.setenv("CC_CLUSTER", "rorqual")
    assert detect_cluster() == "narval"


def test_cc_cluster_env(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.setenv("CC_CLUSTER", "rorqual")
    assert detect_cluster() == "rorqual"


def test_hostname_mila(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.delenv("CC_CLUSTER", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "mila-cluster-login-1")
    assert detect_cluster() == "mila"


def test_unknown_returns_none(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.delenv("CC_CLUSTER", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "my-laptop")
    assert detect_cluster() is None


@pytest.fixture
def _no_env(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.delenv("CC_CLUSTER", raising=False)


@pytest.mark.parametrize(
    "hostname",
    [
        "cn-a001.server.mila.quebec",
        "cn-z099",
        "cn-b042.server.mila.quebec",
    ],
)
def test_detect_mila_compute_cn(monkeypatch, _no_env, hostname):
    monkeypatch.setattr("socket.gethostname", lambda: hostname)
    assert detect_cluster() == "mila"


@pytest.mark.parametrize(
    "hostname",
    [
        "mila-l40s-001",
        "mila-l40s-127.server.mila.quebec",
    ],
)
def test_detect_mila_compute_l40s(monkeypatch, _no_env, hostname):
    monkeypatch.setattr("socket.gethostname", lambda: hostname)
    assert detect_cluster() == "mila"


def test_detect_mila_login(monkeypatch, _no_env):
    monkeypatch.setattr("socket.gethostname", lambda: "mila-cluster-login.server.mila.quebec")
    assert detect_cluster() == "mila"


@pytest.mark.parametrize(
    "hostname",
    [
        "login-1.server.mila.quebec",
        "login-3.server.mila.quebec",
        "login1.server.mila.quebec",
    ],
)
def test_detect_mila_login_server_hostname(monkeypatch, _no_env, hostname):
    monkeypatch.setattr("socket.gethostname", lambda: hostname)
    assert detect_cluster() == "mila"


def test_detect_mila_cpu(monkeypatch, _no_env):
    monkeypatch.setattr("socket.gethostname", lambda: "mila-cpu-001")
    assert detect_cluster() == "mila"


def test_detect_rorqual_login(monkeypatch, _no_env):
    monkeypatch.setattr("socket.gethostname", lambda: "login1.rorqual.alliancecan.ca")
    assert detect_cluster() == "rorqual"


def test_detect_drac_compute_returns_none_without_env(monkeypatch, _no_env):
    monkeypatch.setattr("socket.gethostname", lambda: "nc20101")
    assert detect_cluster() is None


def test_detect_drac_compute_with_cc_cluster_env(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.setenv("CC_CLUSTER", "narval")
    monkeypatch.setattr("socket.gethostname", lambda: "nc20101")
    assert detect_cluster() == "narval"


def test_detect_salvo_cluster_override_wins(monkeypatch):
    monkeypatch.setenv("SALVO_CLUSTER", "foo")
    monkeypatch.setenv("CC_CLUSTER", "rorqual")
    monkeypatch.setattr("socket.gethostname", lambda: "login1.rorqual.alliancecan.ca")
    assert detect_cluster() == "foo"


def test_detect_cc_cluster_beats_hostname(monkeypatch):
    monkeypatch.delenv("SALVO_CLUSTER", raising=False)
    monkeypatch.setenv("CC_CLUSTER", "narval")
    monkeypatch.setattr("socket.gethostname", lambda: "login1.rorqual.alliancecan.ca")
    assert detect_cluster() == "narval"


def test_detect_unknown_host(monkeypatch, _no_env):
    monkeypatch.setattr("socket.gethostname", lambda: "random.example.com")
    assert detect_cluster() is None
