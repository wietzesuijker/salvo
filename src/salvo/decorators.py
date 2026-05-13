"""@cluster.submit decorator. Calls Python entrypoint via PythonEntrypoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from salvo.job.handle import JobHandle
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.job.submit import submit


class _Decorated:
    def __init__(self, fn: Callable[..., Any], spec_kwargs: dict[str, Any], target: str) -> None:
        self.fn = fn
        self.spec_kwargs = spec_kwargs
        self.target = target

    def submit(self, **kwargs: Any) -> JobHandle:
        spec = JobSpec(
            cmd=PythonEntrypoint(target=self.target, kwargs=kwargs),
            **{"name": self.fn.__name__, **self.spec_kwargs},
        )
        return submit(spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class _Cluster:
    def submit(self, **spec_kwargs: Any) -> Callable[[Callable[..., Any]], _Decorated]:
        def decorator(fn: Callable[..., Any]) -> _Decorated:
            target = f"{fn.__module__}:{fn.__qualname__}"
            return _Decorated(fn=fn, spec_kwargs=spec_kwargs, target=target)

        return decorator


cluster = _Cluster()
