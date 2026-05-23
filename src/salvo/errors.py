"""Exception hierarchy. All public exceptions inherit from SalvoError.

The hierarchy is the only thing callers need to catch:

    try:
        salvo.submit(spec)
    except salvo.SalvoError as e:
        ...

Lower-level exceptions raised by subprocess, pydantic, tomllib, etc. are wrapped
at the boundary so the public surface never leaks a stdlib/third-party type.
"""

from __future__ import annotations

import subprocess
from typing import Any


class SalvoError(Exception):
    """Base class for salvo exceptions."""


class ClusterYAMLError(SalvoError):
    """Cluster YAML failed to load or validate."""


class DispatchError(SalvoError):
    """Dispatch (account/partition picking) failed."""


class NoAccountError(DispatchError):
    pass


class NoPartitionError(DispatchError):
    pass


class DataNotStagedError(SalvoError):
    def __init__(
        self,
        cluster: str,
        missing: set[str],
        sources: dict[str, list[str]],
        globus_hint: str | None = None,
    ) -> None:
        self.cluster = cluster
        self.missing = missing
        self.sources = sources
        self.globus_hint = globus_hint
        missing_list = ", ".join(sorted(missing))
        source_hint = "; ".join(
            f"{name}: on {','.join(clusters)}" for name, clusters in sorted(sources.items())
        )
        msg = f"Data not staged on {cluster}: {missing_list}. Sources: {source_hint}"
        if globus_hint:
            msg += f". Hint: {globus_hint}"
        super().__init__(msg)


class MaxHopsExceededError(SalvoError):
    pass


class SubprocessError(SalvoError):
    """Wraps subprocess FileNotFoundError / CalledProcessError / TimeoutExpired.

    Attributes:
        cmd: argv that was attempted.
        returncode: process exit code, or None when the process never ran
            (FileNotFoundError / TimeoutExpired).
        stderr: captured stderr, or "" when unavailable.
    """

    def __init__(
        self,
        message: str,
        *,
        cmd: list[str] | None = None,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.cmd = list(cmd) if cmd is not None else []
        self.returncode = returncode
        self.stderr = stderr


# OomPolicyError, EventSchemaError keep ValueError in their MRO so existing
# `except ValueError:` callers do not regress while migrating to SalvoError.


class OomPolicyError(SalvoError, ValueError):
    """OOM policy DSL failed to parse."""


class EventSchemaError(SalvoError, ValueError):
    """Event name is not in the known taxonomy."""


class ManifestError(SalvoError):
    """Manifest file could not be loaded (corrupt TOML or malformed shape)."""


class SpecValidationError(SalvoError):
    """JobSpec failed pydantic validation at a public entry point."""


def _run_subprocess(
    cmd: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = 10,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and wrap subprocess errors as ``SubprocessError``.

    Catches the three leaky exceptions and re-raises them as ``SubprocessError``
    with ``cmd`` / ``returncode`` / ``stderr`` attached:

    - ``FileNotFoundError``  -> ``{cmd[0]!r} not on PATH``
    - ``CalledProcessError`` -> ``{cmd[0]} failed: {stderr}`` (returncode set)
    - ``TimeoutExpired``     -> ``{cmd[0]} timed out after {timeout}s``

    The happy path passes ``CompletedProcess`` through unchanged.
    """
    bin_name = cmd[0] if cmd else "<empty>"
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            **kwargs,
        )
    except FileNotFoundError as e:
        raise SubprocessError(
            f"{bin_name!r} not on PATH",
            cmd=cmd,
            returncode=None,
            stderr=str(e),
        ) from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        raise SubprocessError(
            f"{bin_name} failed (exit {e.returncode}): {stderr.strip()}",
            cmd=cmd,
            returncode=e.returncode,
            stderr=stderr,
        ) from e
    except subprocess.TimeoutExpired as e:
        raise SubprocessError(
            f"{bin_name} timed out after {timeout}s",
            cmd=cmd,
            returncode=None,
            stderr="",
        ) from e
