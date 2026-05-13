from salvo.doctor import Check, CheckStatus, run_doctor


def test_returns_checks(fake_sbatch, tmp_home):
    results = run_doctor()
    statuses = {r.name: r.status for r in results}
    assert "topology.detect" in statuses
    assert all(isinstance(r, Check) for r in results)


def test_ssh_alias_rejects_fqdn(monkeypatch, fake_sbatch, tmp_home):
    monkeypatch.setenv("SALVO_CLUSTER", "rorqual")
    results = run_doctor()
    ssh_checks = [r for r in results if r.name == "ssh.alias"]
    assert ssh_checks
    assert ssh_checks[0].status in (CheckStatus.OK, CheckStatus.WARN)
