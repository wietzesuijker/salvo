"""Public history surface for memory-usage learning.

Mirrors ``salvo.policy``: external callers (cluv, xgenius, user code) import
from here; the implementation lives in ``salvo.job.history``.
"""

from __future__ import annotations

from salvo.job.history import (
    DEGENERATE_RSS_RATIO,
    LEARNABLE_STATES,
    Confidence,
    JobRecord,
    MemEstimate,
    estimate_mem,
    format_suggestion,
    spec_key,
)

__all__ = [
    "DEGENERATE_RSS_RATIO",
    "LEARNABLE_STATES",
    "Confidence",
    "JobRecord",
    "MemEstimate",
    "estimate_mem",
    "format_suggestion",
    "spec_key",
]
