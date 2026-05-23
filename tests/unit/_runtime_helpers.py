"""Helper targets imported by test_runtime_entrypoint via `tests.unit._runtime_helpers`."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def simple(**kw: Any) -> dict[str, Any]:
    return kw


def raises_runtime(**kw: Any) -> None:
    raise RuntimeError("boom")


class Nested:
    @staticmethod
    def fn(**kw: Any) -> dict[str, Any]:
        return kw


def write_marker(path: str, **kw: Any) -> None:
    Path(path).write_text("ok")


NOT_CALLABLE = 42
