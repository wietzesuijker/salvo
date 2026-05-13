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
