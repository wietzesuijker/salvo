"""Cluster topology pydantic models. schema_version=1 is the stable surface."""

from __future__ import annotations

import operator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_COMPARATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
}


class Account(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    gpu_cap: int = 0
    cpu_cap: int = 0
    mem_cap_gb: int = 0
    gpu_only: bool = False  # no _cpu partner exists


class Partition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    max_walltime_hours: int
    max_gpus_per_job: int = 0
    max_mem_gb: int | None = None
    requires_account_suffix: Literal["_cpu", "_gpu", None] = None
    prefer_mem_kind: Literal["total", "per-gpu"] = "total"
    requires_qos: str | None = None


class DispatchRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    when: dict[str, Any] = Field(default_factory=dict)
    prefer: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)
    reason: str = ""

    def matches(self, *, gpus: int = 0, cpus: int = 0, mem_mb: int = 0) -> bool:
        ctx = {"gpus": gpus, "cpus": cpus, "mem_mb": mem_mb}
        for key, predicate in self.when.items():
            if key not in ctx:
                return False
            actual = ctx[key]
            if isinstance(predicate, str) and predicate[:2] in _COMPARATORS:
                op = _COMPARATORS[predicate[:2]]
                rhs = float(predicate[2:].strip())
            elif isinstance(predicate, str) and predicate[:1] in _COMPARATORS:
                op = _COMPARATORS[predicate[:1]]
                rhs = float(predicate[1:].strip())
            else:
                op = operator.eq
                rhs = predicate
            if not op(actual, rhs):
                return False
        return True


class LoginConstraints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    ssh_alias: str
    apptainer_env: dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    scratch_root: str = ""
    project_root: str = ""


class QoSUser(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_cpus: int = 0
    max_mem_gb: int = 0


class StorageHints(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    prefer_for_outputs: str = ""
    symlink_eval_outputs: bool = False


class Cluster(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int
    id: str
    display_name: str
    type: Literal["national", "local", "slurm"]
    extends: str | None = None
    gpu_model: str | None = None
    cpus_per_gpu: int = 0
    mem_per_gpu_gb: int = 0
    has_internet: bool = True
    login: LoginConstraints | None = None
    accounts: list[Account] = Field(default_factory=list)
    partitions: list[Partition] = Field(default_factory=list)
    qos_user: QoSUser | None = None
    dispatch_rules: list[DispatchRule] = Field(default_factory=list)
    data: DataConfig = Field(default_factory=DataConfig)
    container_paths: list[str] = Field(default_factory=list)
    storage_hints: StorageHints = Field(default_factory=StorageHints)
