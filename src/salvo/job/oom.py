"""OOM policy DSL parser + retry. Phase 0 implements bump_mem + escalate_partition + fail.
Callback and bump_gpus parse but are NOT applied (Phase 2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import dataclass as _dc
from typing import Literal, NamedTuple

from salvo.job.spec import JobSpec, parse_mem_mb

Action = Literal["bump_mem", "escalate_partition", "fail"]
Kind = Literal["cpu", "gpu"]


class OomDecision(NamedTuple):
    """Result of ``apply_oom``. Tuple-unpacks for back-compat with ``(spec, action)``.

    ``new_spec`` is ``None`` when the policy ran out of steps (terminal fail).
    ``action`` is the literal name of the step that fired.
    """

    new_spec: JobSpec | None
    action: Action


@dataclass(frozen=True)
class BumpMemStep:
    factor: float
    max_mb: int


@dataclass(frozen=True)
class EscalatePartitionStep:
    pass


@dataclass(frozen=True)
class BumpGpusStep:
    factor: float
    max_gpus: int


@dataclass(frozen=True)
class CallbackStep:
    target: str


@dataclass(frozen=True)
class FailStep:
    pass


Step = BumpMemStep | EscalatePartitionStep | BumpGpusStep | CallbackStep | FailStep


_BUMP_MEM_RE = re.compile(r"^bump_mem\(\s*(\d+(?:\.\d+)?)x\s*,\s*max=([^)]+?)\s*\)$")
_BUMP_GPUS_RE = re.compile(r"^bump_gpus\(\s*(\d+(?:\.\d+)?)x\s*,\s*max=(\d+)\s*\)$")
_CALLBACK_RE = re.compile(r"^callback\(\s*([\w.:]+)\s*\)$")


def _parse_one(s: str) -> Step:
    s = s.strip()
    if m := _BUMP_MEM_RE.match(s):
        return BumpMemStep(factor=float(m.group(1)), max_mb=parse_mem_mb(m.group(2)))
    if s == "escalate_partition":
        return EscalatePartitionStep()
    if m := _BUMP_GPUS_RE.match(s):
        return BumpGpusStep(factor=float(m.group(1)), max_gpus=int(m.group(2)))
    if m := _CALLBACK_RE.match(s):
        target = m.group(1)
        if ":" not in target:
            raise ValueError(f"callback target must be module.path:function, got {target!r}")
        return CallbackStep(target=target)
    if s == "fail":
        return FailStep()
    raise ValueError(f"unknown on_oom step: {s!r}")


def parse_policy(steps: list[str]) -> list[Step]:
    if not steps:
        return [FailStep()]
    return [_parse_one(s) for s in steps]


# --- OOM retry ---


@_dc(frozen=True)
class OomContext:
    kind: Kind
    max_rss_mb: int | None = None
    log_excerpt: str = ""


def apply_oom_policy(spec: JobSpec, ctx: OomContext) -> OomDecision:
    """Apply policy steps in order; return ``OomDecision(new_spec, action)``.

    Step semantics today:

    - ``bump_mem(<factor>x, max=<size>)`` fires only when ``ctx.kind == "cpu"``.
      On ``kind == "gpu"`` it falls through so the policy can route to a
      GPU-aware step (typically ``escalate_partition`` or ``fail``).
    - ``escalate_partition`` clears ``spec.partition`` so the next render
      re-picks a larger tier.
    - ``bump_gpus`` and ``callback`` parse today and fall through (planned for
      a later release).
    - ``fail`` is terminal and returns ``OomDecision(None, "fail")``.

    The first applicable step wins; if no step applies, the result is
    ``OomDecision(None, "fail")``.
    """
    for step in parse_policy(spec.on_oom):
        if isinstance(step, BumpMemStep) and ctx.kind == "cpu":
            current_mb = spec.mem_mb()
            target = max(int(current_mb * step.factor), int((ctx.max_rss_mb or current_mb) * 1.2))
            new_mb = min(target, step.max_mb)
            if new_mb <= current_mb:
                continue
            return OomDecision(spec.model_copy(update={"mem": f"{new_mb}M"}), "bump_mem")
        if isinstance(step, EscalatePartitionStep):
            # Phase 0: leave partition None to trigger re-pick on resubmit
            if spec.partition is not None:
                return OomDecision(
                    spec.model_copy(update={"partition": None}), "escalate_partition"
                )
            continue
        if isinstance(step, BumpGpusStep | CallbackStep):
            # Phase 0: parse-only; treat as fall-through
            continue
        if isinstance(step, FailStep):
            return OomDecision(None, "fail")
    return OomDecision(None, "fail")
