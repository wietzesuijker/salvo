"""JobSpec: immutable input to submit(). Pure function from spec + cluster → sbatch.sh."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MEM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMG]B?)\s*$", re.IGNORECASE)
_HMS_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d+):(\d+)$")
_HUMAN_TIME_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE)


def parse_mem_mb(s: str) -> int:
    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"invalid memory string: {s!r}")
    value, unit = float(m.group(1)), m.group(2).upper().rstrip("B")
    multiplier = {"K": 1 / 1024, "M": 1, "G": 1024}[unit]
    return max(1, round(value * multiplier))


def parse_time_min(s: str) -> int:
    s = s.strip()
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
