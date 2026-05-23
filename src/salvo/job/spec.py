"""JobSpec: immutable input to submit(). Pure function from spec + cluster → sbatch.sh."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Memory grammar accepts K/M/G/T with optional ``B`` or ``iB`` (e.g. ``GiB``).
# All units treated as binary (1024-power) to match SLURM convention.
_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT])(?:I?B)?\s*$", re.IGNORECASE)
_HMS_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d+):(\d+)$")
_HUMAN_TIME_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE)

# Shell-safe env var keys: POSIX-style identifier.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_mem_mb(s: str) -> int:
    """Parse a memory string to MB.

    Accepts K/M/G/T with optional ``B`` or ``iB`` suffix (e.g. ``"4G"``,
    ``"4GiB"``, ``"4GB"``, ``"1.5T"``). All units are treated as binary
    (1024-power) to match SLURM's convention for bare ``M``/``G``/``T``.
    """
    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"invalid memory string: {s!r}")
    value, unit = float(m.group(1)), m.group(2).upper()
    multiplier = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    return max(1, round(value * multiplier))


def parse_time_min(s: str) -> int:
    """Parse a SLURM walltime string to minutes (minimum 1).

    Accepts ``D-HH:MM:SS``, ``HH:MM:SS``, ``1h30m``, ``45s``, or a bare integer
    interpreted as minutes (e.g. ``"30"`` → 30). Rejects ``"0"`` explicitly:
    SLURM treats ``--time=0`` as unlimited, which is too dangerous a default.
    """
    s = s.strip()
    if s == "0":
        raise ValueError(
            "time='0' (unlimited) not supported; use a finite walltime "
            "or set partition timelimit"
        )
    if s.isdigit():
        return max(1, int(s))
    m = _HMS_RE.match(s)
    if m:
        days, h, mins, sec = (int(x or 0) for x in m.groups())
        return max(1, days * 1440 + h * 60 + mins + (1 if sec else 0))
    m = _HUMAN_TIME_RE.match(s)
    if m and any(m.groups()):
        h, mins, sec = (int(x or 0) for x in m.groups())
        return max(1, h * 60 + mins + (1 if sec else 0))
    raise ValueError(f"invalid time string: {s!r}")


class PythonEntrypoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    target: str  # "module.path:function"
    kwargs: dict[str, object] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        if ":" not in v or v.count(":") != 1:
            raise ValueError("target must be 'module.path:function'")
        mod, fn = v.split(":")
        if not mod or not fn:
            raise ValueError("target module and function must be non-empty")
        return v


class Hop(BaseModel):
    model_config = ConfigDict(frozen=True)
    parent_job_id: str
    child_job_id: str
    cause: Literal["oom_cpu", "oom_gpu", "preempt", "max_hops_exceeded"]
    new_mem_mb: int | None = None
    new_partition: str | None = None
    hop_index: int


class JobStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    job_id: str
    state: str  # PENDING | RUNNING | COMPLETED | FAILED | CANCELLED | OUT_OF_MEMORY | TIMEOUT
    exit_code: int | None = None
    elapsed_min: int | None = None
    max_rss_mb: int | None = None
    max_rss_kb: int | None = None


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    cmd: list[str] | PythonEntrypoint
    gpus: int = Field(default=0, ge=0)
    cpus: int = Field(default=1, ge=1)
    mem: str = "4G"
    time: str = "1h"
    partition: str | None = None
    account: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    data_needs: list[str] = Field(default_factory=list)
    data_optional: list[str] = Field(default_factory=list)
    container: str | None = None
    on_oom: list[str] = Field(default_factory=list)
    on_preempt: Literal["resubmit", "fail"] = "resubmit"
    max_hops: int = Field(default=5, ge=0, le=20)
    deps: list[str] = Field(default_factory=list)  # job_id strings of dependencies

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not v:
            raise ValueError("name must be non-empty")
        # Reject newlines, NUL, and any other control character. The name is
        # interpolated raw into ``#SBATCH --job-name={name}``; a newline would
        # allow injecting arbitrary directives.
        for ch in v:
            if ch == "\n" or ch == "\r" or ch == "\x00" or not ch.isprintable():
                raise ValueError(f"name must not contain control characters; got {v!r}")
        return v

    @field_validator("env")
    @classmethod
    def _check_env(cls, v: dict[str, str]) -> dict[str, str]:
        # Env keys become shell ``export`` identifiers; restrict to POSIX form.
        for k in v:
            if not _ENV_KEY_RE.match(k):
                raise ValueError(f"env key must match [A-Za-z_][A-Za-z0-9_]*; got {k!r}")
        return v

    @field_validator("mem")
    @classmethod
    def _check_mem(cls, v: str) -> str:
        parse_mem_mb(v)
        return v

    @field_validator("time")
    @classmethod
    def _check_time(cls, v: str) -> str:
        parse_time_min(v)
        return v

    def mem_mb(self) -> int:
        return parse_mem_mb(self.mem)

    def time_min(self) -> int:
        return parse_time_min(self.time)
