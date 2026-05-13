from salvo import cluster


def test_decorator_captures_spec(fake_sbatch, tmp_home):
    @cluster.submit(gpus=0, cpus=2, mem="4G", time="30m", name="dec_test")
    def hello(seed: int): ...

    h = hello.submit(seed=1)
    assert h.job_id.isdigit()
    assert h.spec.name == "dec_test"
    assert h.spec.cpus == 2
