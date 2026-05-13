"""Cap-aware account picker with dispatch_rules engine."""

from __future__ import annotations

from salvo.dispatch.caps import CapsSnapshot
from salvo.errors import NoAccountError
from salvo.job.spec import JobSpec
from salvo.topology.schema import Account, Cluster


def _has_capacity(acct: Account, snap: CapsSnapshot, *, gpus: int, cpus: int, mem_mb: int) -> bool:
    if gpus > acct.gpu_cap - snap.gpus_in_use.get(acct.name, 0):
        return False
    if cpus > acct.cpu_cap - snap.cpus_in_use.get(acct.name, 0):
        return False
    return not (mem_mb > (acct.mem_cap_gb * 1024) - snap.mem_in_use_mb.get(acct.name, 0))


def pick_account(spec: JobSpec, cluster: Cluster, snap: CapsSnapshot) -> str:
    if spec.account:
        if not any(a.name == spec.account for a in cluster.accounts):
            raise NoAccountError(f"explicit account {spec.account!r} not in cluster {cluster.id}")
        return spec.account

    gpus, cpus, mem_mb = spec.gpus, spec.cpus, spec.mem_mb()
    candidates: list[str] = []
    forbidden: set[str] = set()
    for rule in cluster.dispatch_rules:
        if rule.matches(gpus=gpus, cpus=cpus, mem_mb=mem_mb):
            for a in rule.prefer:
                if a not in candidates:
                    candidates.append(a)
            forbidden |= set(rule.forbid)

    if not candidates:
        candidates = [a.name for a in cluster.accounts]

    accounts_by_name = {a.name: a for a in cluster.accounts}
    for name in candidates:
        if name in forbidden or name not in accounts_by_name:
            continue
        acct = accounts_by_name[name]
        if acct.gpu_only and gpus == 0:
            continue
        if _has_capacity(acct, snap, gpus=gpus, cpus=cpus, mem_mb=mem_mb):
            return name

    raise NoAccountError(
        f"no eligible account on {cluster.id} for spec "
        f"(gpus={gpus}, cpus={cpus}, mem_mb={mem_mb}). "
        f"Tried: {candidates}. Forbidden: {sorted(forbidden)}."
    )
