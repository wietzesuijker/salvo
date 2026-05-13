"""Public policy surface — thin re-exports over salvo.job.oom with cleaner names.

External callers (cluv, xgenius, user code) should import from here:

    from salvo.policy import parse, apply_oom, Step, OomContext
    from salvo.policy import BumpMemStep, EscalatePartitionStep, FailStep

The implementation lives in salvo.job.oom; this module only renames
`parse_policy` -> `parse` and `apply_oom_policy` -> `apply_oom`.
"""

from __future__ import annotations

from salvo.job.oom import (
    BumpGpusStep,
    BumpMemStep,
    CallbackStep,
    EscalatePartitionStep,
    FailStep,
    OomContext,
    Step,
)
from salvo.job.oom import (
    apply_oom_policy as apply_oom,
)
from salvo.job.oom import (
    parse_policy as parse,
)

__all__ = [
    "BumpGpusStep",
    "BumpMemStep",
    "CallbackStep",
    "EscalatePartitionStep",
    "FailStep",
    "OomContext",
    "Step",
    "apply_oom",
    "parse",
]
