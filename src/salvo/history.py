"""Public history surface for memory-usage learning.

Mirrors ``salvo.policy``: external callers (cluv, xgenius, user code) import
from here; the implementation lives in ``salvo.job.history``.
"""

from __future__ import annotations

from salvo.job.history import (
    LEARNABLE_STATES,
    Confidence,
    JobRecord,
    MemEstimate,
    estimate_mem,
    spec_key,
)

__all__ = [
    "LEARNABLE_STATES",
    "Confidence",
    "JobRecord",
    "MemEstimate",
    "estimate_mem",
    "spec_key",
]
