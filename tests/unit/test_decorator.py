import inspect

import pytest
from salvo import cluster


def _hello(seed: int): ...


def test_decorator_captures_spec(fake_sbatch, tmp_home):
    decorated = cluster.submit(gpus=0, cpus=2, mem="4G", time="30m", name="dec_test")(_hello)
    h = decorated.submit(seed=1)
    assert h.job_id.isdigit()
    assert h.spec.name == "dec_test"
    assert h.spec.cpus == 2


def _docstring_fn(x: int) -> None:
    """A real module-level function with a docstring."""


def test_preserves_metadata():
    decorated = cluster.submit(cpus=1, mem="1G", time="5m")(_docstring_fn)
    assert decorated.__name__ == "_docstring_fn"
    assert decorated.__doc__ == "A real module-level function with a docstring."
    assert decorated.__module__ == _docstring_fn.__module__
    assert decorated.__qualname__ == _docstring_fn.__qualname__
    assert decorated.__wrapped__ is _docstring_fn


def _typed_fn(x: int) -> None:
    return None


def test_typed_signature():
    decorated = cluster.submit(cpus=1, mem="1G", time="5m")(_typed_fn)
    sig = inspect.signature(decorated)
    assert "x" in sig.parameters
    assert sig.parameters["x"].annotation is int
    assert sig.return_annotation is None


def _two_arg_fn(x: int, y: int = 2) -> int:
    return x + y


def test_submit_validates_kwargs(fake_sbatch, tmp_home):
    decorated = cluster.submit(cpus=1, mem="1G", time="5m")(_two_arg_fn)
    with pytest.raises(TypeError) as exc_info:
        decorated.submit(x=1, typo=1)
    assert "typo" in str(exc_info.value) or "unexpected" in str(exc_info.value).lower()


def _seed_fn(seed: object) -> None: ...


def test_submit_json_serializable(fake_sbatch, tmp_home):
    decorated = cluster.submit(cpus=1, mem="1G", time="5m")(_seed_fn)
    with pytest.raises(ValueError, match="JSON-serializable"):
        decorated.submit(seed={1, 2, 3})


def test_rejects_closure():
    def outer():
        def inner(x: int) -> None: ...

        return inner

    inner = outer()
    with pytest.raises(ValueError, match="module-level"):
        cluster.submit(cpus=1, mem="1G", time="5m")(inner)


def test_rejects_lambda():
    f = lambda x: x  # noqa: E731
    with pytest.raises(ValueError, match="module-level"):
        cluster.submit(cpus=1, mem="1G", time="5m")(f)


def test_rejects_cmd_kwarg():
    with pytest.raises(ValueError, match="cmd"):
        cluster.submit(cmd=["echo", "hi"], cpus=1, mem="1G", time="5m")


def test_call_passthrough_preserves_return():
    decorated = cluster.submit(cpus=1, mem="1G", time="5m")(_two_arg_fn)
    assert decorated(3, 4) == 7
    assert decorated(3) == 5  # default y=2
