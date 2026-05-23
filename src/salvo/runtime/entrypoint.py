"""JSON-payload → callable shim run on the compute node by the rendered sbatch script.

Invoked as::

    python -m salvo.runtime.entrypoint '{"target":"pkg.mod:fn","kwargs":{...}}'

Contract:

* Exit 0 on success.
* Exit 2 on any payload error (bad JSON, missing keys, wrong types, unresolvable
  target, non-callable resolved object).
* The wrapped callable's own exceptions propagate unchanged so SLURM marks the
  job FAILED with the natural Python exit code.

Stdlib only. No imports from ``salvo.*`` — the wrapped callable may live in a
totally different repo and this shim must remain dependency-free.
"""

from __future__ import annotations

import functools
import importlib
import json
import sys
from typing import Any

_USAGE = (
    "usage: python -m salvo.runtime.entrypoint "
    '\'{"target":"module.path:attribute","kwargs":{...}}\''
)


def parse_payload(raw: str) -> tuple[str, str, dict[str, Any]]:
    """Parse the JSON payload string into ``(module, attr_path, kwargs)``.

    Raises ``ValueError`` on any shape problem. Caller is responsible for
    translating that into the exit-2 stderr contract.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON payload: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"payload must be a JSON object, got {type(obj).__name__}")
    if "target" not in obj:
        raise ValueError("missing 'target' key in payload")
    if "kwargs" not in obj:
        raise ValueError("missing 'kwargs' key in payload")
    target = obj["target"]
    kwargs = obj["kwargs"]
    if not isinstance(target, str):
        raise ValueError(f"'target' must be str, got {type(target).__name__}")
    if not isinstance(kwargs, dict):
        raise ValueError(f"'kwargs' must be a dict, got {type(kwargs).__name__}")
    if target.count(":") != 1:
        raise ValueError(f"target must be module.path:attribute, got {target!r}")
    module_part, attr_part = target.split(":")
    if not module_part or not attr_part:
        raise ValueError(f"target must be module.path:attribute, got {target!r}")
    return module_part, attr_part, kwargs


def resolve_target(module_part: str, attr_part: str) -> Any:
    """Import ``module_part`` and walk ``attr_part`` (dotted) on it.

    Raises ``ImportError`` if the module can't be imported, ``AttributeError``
    if any attribute lookup along the path fails, ``TypeError`` if the final
    object isn't callable.
    """
    try:
        module = importlib.import_module(module_part)
    except ImportError as e:
        raise ImportError(f"cannot import module {module_part!r}: {e}") from e
    parts = attr_part.split(".")
    current: Any = module
    walked: list[str] = []
    for part in parts:
        try:
            current = functools.reduce(getattr, [part], current)
        except AttributeError as e:
            owner = ".".join([module_part, *walked]) if walked else module_part
            raise AttributeError(f"attribute {part!r} not found on module {owner!r}") from e
        walked.append(part)
    if not callable(current):
        raise TypeError(
            f"{module_part}.{attr_part!r} is not callable " f"(got {type(current).__name__})"
        )
    return current


def _fail(msg: str) -> None:
    """Print a payload-error message to stderr and exit with code 2."""
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    """Entry point. ``argv`` excludes the program name (mirrors ``sys.argv[1:]``)."""
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        _fail(_USAGE)
    try:
        module_part, attr_part, kwargs = parse_payload(argv[0])
    except ValueError as e:
        _fail(str(e))
        return  # unreachable; appeases mypy
    try:
        target = resolve_target(module_part, attr_part)
    except (ImportError, AttributeError, TypeError) as e:
        _fail(str(e))
        return  # unreachable
    # User-callable exceptions propagate intentionally.
    target(**kwargs)


if __name__ == "__main__":
    main()
