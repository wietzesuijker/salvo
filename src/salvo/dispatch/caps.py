"""Per-process squeue snapshot + in-process counter for sub-snapshot accuracy."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class CapsSnapshot:
    gpus_in_use: dict[str, int] = field(default_factory=dict)
    cpus_in_use: dict[str, int] = field(default_factory=dict)
    mem_in_use_mb: dict[str, int] = field(default_factory=dict)


def _run_squeue(user: str) -> str:
    out = subprocess.run(
        ["squeue", "-u", user, "-o", "%i|%a|%P|%b|%C|%m", "-h"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return out.stdout


def _parse_mem(field_str: str) -> int:
    if not field_str:
        return 0
    f = field_str.rstrip("MGT").rstrip("KMG")
    try:
        return int(float(f))
    except ValueError:
        return 0


class CapsTracker:
    def __init__(self, cluster_id: str, user: str, ttl_sec: int = 60) -> None:
        self.cluster_id = cluster_id
        self.user = user
        self.ttl_sec = ttl_sec
        self._snapshot: CapsSnapshot | None = None
        self._snapshot_at: float = 0.0
        self._delta_gpus: dict[str, int] = {}
        self._delta_cpus: dict[str, int] = {}
        self._delta_mem: dict[str, int] = {}

    def _refresh(self) -> None:
        text = _run_squeue(self.user)
        snap = CapsSnapshot()
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            _job, acct, _part, gres, cpus, mem = parts[:6]
            n_gpus = 0
            if "gpu:" in gres:
                try:
                    n_gpus = int(gres.split(":")[-1])
                except ValueError:
                    n_gpus = 1
            snap.gpus_in_use[acct] = snap.gpus_in_use.get(acct, 0) + n_gpus
            snap.cpus_in_use[acct] = snap.cpus_in_use.get(acct, 0) + int(cpus or 0)
            snap.mem_in_use_mb[acct] = snap.mem_in_use_mb.get(acct, 0) + _parse_mem(mem)
        self._snapshot = snap
        self._snapshot_at = time.monotonic()
        self._delta_gpus.clear()
        self._delta_cpus.clear()
        self._delta_mem.clear()

    def snapshot(self) -> CapsSnapshot:
        if self._snapshot is None or time.monotonic() - self._snapshot_at >= self.ttl_sec:
            saved_gpus = dict(self._delta_gpus)
            saved_cpus = dict(self._delta_cpus)
            saved_mem = dict(self._delta_mem)
            self._refresh()
            self._delta_gpus = saved_gpus
            self._delta_cpus = saved_cpus
            self._delta_mem = saved_mem
        assert self._snapshot is not None
        combined = CapsSnapshot()
        for acct, v in self._snapshot.gpus_in_use.items():
            combined.gpus_in_use[acct] = v + self._delta_gpus.get(acct, 0)
        for acct, v in self._delta_gpus.items():
            combined.gpus_in_use.setdefault(acct, v)
        for acct, v in self._snapshot.cpus_in_use.items():
            combined.cpus_in_use[acct] = v + self._delta_cpus.get(acct, 0)
        for acct, v in self._delta_cpus.items():
            combined.cpus_in_use.setdefault(acct, v)
        for acct, v in self._snapshot.mem_in_use_mb.items():
            combined.mem_in_use_mb[acct] = v + self._delta_mem.get(acct, 0)
        for acct, v in self._delta_mem.items():
            combined.mem_in_use_mb.setdefault(acct, v)
        return combined

    def account_after_submit(self, account: str, *, gpus: int, cpus: int, mem_mb: int) -> None:
        self._delta_gpus[account] = self._delta_gpus.get(account, 0) + gpus
        self._delta_cpus[account] = self._delta_cpus.get(account, 0) + cpus
        self._delta_mem[account] = self._delta_mem.get(account, 0) + mem_mb
