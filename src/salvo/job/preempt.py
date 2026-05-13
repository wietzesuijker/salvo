"""Preempt wrapper logic: invoked from sbatch epilog. Pure functions (testable)."""

from __future__ import annotations

from typing import Literal

_DRAC_SUFFIXES = ("_gpu", "_cpu")


def strip_account_suffix(account: str) -> str:
    for suffix in _DRAC_SUFFIXES:
        if account.endswith(suffix):
            return account[: -len(suffix)]
    return account


def next_hop(salvo_hop_env: str) -> tuple[str, bool]:
    """Return (new SALVO_HOP value, max_exceeded)."""
    current_s, max_s = salvo_hop_env.split("/")
    current, mx = int(current_s), int(max_s)
    if current >= mx:
        return salvo_hop_env, True
    return f"{current + 1}/{mx}", False


def should_resubmit(
    *, hop_env: str, on_preempt: Literal["resubmit", "fail"], artifact_changed: bool
) -> bool:
    if on_preempt != "resubmit":
        return False
    if not artifact_changed:
        return False
    _, max_exceeded = next_hop(hop_env)
    return not max_exceeded
