"""@cluster.submit decorator. Calls Python entrypoint via PythonEntrypoint.

Hardens the decorator with:
- functools.update_wrapper (preserves __name__, __doc__, __module__, __qualname__, __wrapped__)
- typed signature preservation via ParamSpec + TypeVar so mypy --strict consumers
  (e.g. cluv) keep typed args + return on both __call__ and .submit
- inspect.signature(...).bind(**kwargs) validation at .submit() time
- rejection of closures, lambdas, and locals-defined functions at decoration time
- json.dumps round-trip validation of kwargs before constructing JobSpec
- rejection of cmd= passed into spec_kwargs (avoids JobSpec arg collision)
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar

from salvo.job.handle import JobHandle
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.job.submit import submit

P = ParamSpec("P")
R = TypeVar("R")


class DecoratedFn(Protocol[P, R]):
    """Protocol describing the object returned by @cluster.submit(...).

    Preserves the wrapped function's signature for direct calls and offers a
    typed .submit(**kwargs) that returns a JobHandle.
    """

    __wrapped__: Callable[P, R]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> JobHandle: ...


def _reject_unimportable(fn: Callable[..., Any]) -> None:
    """Reject functions whose target cannot be imported on a worker.

    Lambdas, closures, and locals-defined functions render to a target like
    ``mod:outer.<locals>.inner`` which no importer can resolve. Catch this at
    decoration time so users get a clear error instead of a worker-side crash.
    """
    qualname = getattr(fn, "__qualname__", "")
    name = getattr(fn, "__name__", "")
    if name == "<lambda>" or "<locals>" in qualname:
        raise ValueError(
            "@cluster.submit only supports module-level functions; " f"got {qualname or name!r}"
        )


class _Decorated:
    """Concrete decorated wrapper. Use DecoratedFn for type annotations."""

    def __init__(
        self,
        fn: Callable[..., Any],
        spec_kwargs: dict[str, Any],
        target: str,
    ) -> None:
        self.fn = fn
        self.spec_kwargs = spec_kwargs
        self.target = target
        self._sig = inspect.signature(fn)
        functools.update_wrapper(self, fn)

    def submit(self, *args: Any, **kwargs: Any) -> JobHandle:
        # 1. Validate kwargs against the wrapped function's signature so a typo
        #    like seeed=1 fails locally rather than on the worker.
        try:
            bound = self._sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{self.fn.__name__}() submit-kwargs mismatch: {e}") from e
        bound.apply_defaults()
        call_kwargs = dict(bound.arguments)

        # 2. Validate kwargs round-trip through json (PythonEntrypoint.kwargs is
        #    rendered to JSON downstream).
        try:
            json.dumps(call_kwargs)
        except TypeError as e:
            raise ValueError(f"kwargs must be JSON-serializable: {e}") from e

        spec = JobSpec(
            cmd=PythonEntrypoint(target=self.target, kwargs=call_kwargs),
            **{"name": self.fn.__name__, **self.spec_kwargs},
        )
        return submit(spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class _Cluster:
    def submit(self, **spec_kwargs: Any) -> Callable[[Callable[P, R]], DecoratedFn[P, R]]:
        if "cmd" in spec_kwargs:
            raise ValueError(
                "@cluster.submit does not accept cmd= " "(use the decorated function's body)"
            )

        def decorator(fn: Callable[P, R]) -> DecoratedFn[P, R]:
            _reject_unimportable(fn)
            target = f"{fn.__module__}:{fn.__qualname__}"
            wrapper = _Decorated(fn=fn, spec_kwargs=spec_kwargs, target=target)
            # _Decorated is structurally a DecoratedFn[P, R]; cast for the type
            # system without changing runtime behavior.
            from typing import cast

            return cast(DecoratedFn[P, R], wrapper)

        return decorator


cluster = _Cluster()
