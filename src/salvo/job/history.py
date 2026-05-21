"""Memory-usage learning over a window of past terminal jobs.

Pure functions. cluv (or any front-end) collects ``JobRecord``s from sacct and
calls ``estimate_mem`` to get a memory ask. Right-censoring on OOM rows, growth
detection, and confidence scoring are handled here so front-ends stay thin.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from statistics import median
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

# States from which we can extract a memory observation. Anything outside this
# set (PENDING, RUNNING, CANCELLED-before-start, ...) is dropped.
LEARNABLE_STATES = frozenset({"COMPLETED", "FAILED", "OUT_OF_MEMORY"})

Confidence = Literal["high", "medium", "low", "none"]


class JobRecord(BaseModel):
    """One terminal-state job observation usable by the estimator.

    Persisted by cluv in its local cache; consumed by ``estimate_mem`` here.
    """

    model_config = ConfigDict(frozen=True)

    job_id: str
    key: str
    cluster: str
    state: str
    mem_mb: int
    max_rss_mb: int | None = None
    elapsed_s: int | None = None
    submitted_at: datetime


class MemEstimate(NamedTuple):
    """Output of ``estimate_mem``.

    ``mem_mb`` is ``None`` when there is not enough history; the caller should
    then fall back to its configured default.
    """

    mem_mb: int | None
    confidence: Confidence
    n_samples: int
    p95_mb: int | None
    growth_slope_mb_per_run: float | None
    rationale: str


def spec_key(
    script_path: str,
    git_commit: str | None,
    program_args: tuple[str, ...] = (),
) -> str:
    """Stable identifier for "the same job" across submissions.

    Includes ``git_commit`` so a code change resets the history; pre-refactor
    memory profiles should not bind post-refactor code. When ``git_commit`` is
    ``None``, only the path and args participate.
    """
    h = hashlib.blake2s(digest_size=12)
    h.update(script_path.encode())
    h.update(b"\0")
    h.update((git_commit or "").encode())
    h.update(b"\0")
    for arg in program_args:
        h.update(arg.encode())
        h.update(b"\0")
    return h.hexdigest()


def _observation(record: JobRecord) -> int | None:
    """Map a record to a single memory-used number, handling right-censoring.

    - ``COMPLETED`` with a ``max_rss_mb``: that value is the true upper bound.
    - ``OUT_OF_MEMORY``: ``MaxRSS`` is truncated at the request, so use
      ``mem_mb`` as a lower-bound observation. A later successful run will
      raise the ceiling; until then the estimator at least matches the last
      OOM ask, which is better than under-predicting.
    - Anything without ``max_rss_mb`` and not OOM: not informative, drop.
    """
    if record.state == "OUT_OF_MEMORY":
        return max(record.mem_mb, record.max_rss_mb or 0)
    if record.max_rss_mb is not None and record.max_rss_mb > 0:
        return record.max_rss_mb
    return None


def _percentile(values: list[int], p: int) -> int:
    """Empirical percentile that rounds up on a fractional rank.

    Conservative for small samples: P95 of three values is the largest.
    """
    if not values:
        return 0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return max(s[lo], s[hi])


def _slope(values: list[int]) -> float:
    """Ordinary least-squares slope of ``values`` against their index.

    Returns 0.0 for fewer than two points. Used to flag a monotone upward
    drift in MaxRSS across consecutive runs.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def estimate_mem(
    records: list[JobRecord],
    *,
    safety: float = 1.2,
    window: int = 20,
    min_samples: int = 3,
    growth_bump: float = 1.25,
    growth_threshold_pct: float = 0.05,
) -> MemEstimate:
    """Estimate a memory ask from a window of past ``JobRecord``s.

    1. Take the newest ``window`` records, map each to a learnable observation.
    2. If fewer than ``min_samples`` observations survive, return
       ``MemEstimate(None, "none", ...)``; the caller should fall back to its
       configured default.
    3. Compute P95 of the observations, multiply by ``safety``.
    4. If the chronological series shows an upward drift greater than
       ``growth_threshold_pct`` of the median per run, multiply by
       ``growth_bump`` so the estimate leans toward where the trend is going.
    5. Confidence is ``"high"`` for ``n >= 2 * min_samples`` and stable,
       ``"medium"`` when sufficient but small or growing, ``"low"`` otherwise.
    """
    if not records:
        return MemEstimate(None, "none", 0, None, None, "no history records")

    recent = sorted(records, key=lambda r: r.submitted_at, reverse=True)[:window]
    observations: list[tuple[datetime, int]] = []
    for record in recent:
        if record.state not in LEARNABLE_STATES:
            continue
        obs = _observation(record)
        if obs is not None:
            observations.append((record.submitted_at, obs))

    n = len(observations)
    if n < min_samples:
        return MemEstimate(
            None,
            "none",
            n,
            None,
            None,
            f"only {n} usable observation(s); need >= {min_samples}",
        )

    values = [v for _, v in observations]
    p95 = _percentile(values, 95)
    chronological = [v for _, v in sorted(observations, key=lambda x: x[0])]
    slope = _slope(chronological) if n >= 4 else 0.0
    med = median(values)
    growing = med > 0 and slope > growth_threshold_pct * med
    effective_safety = safety * (growth_bump if growing else 1.0)
    estimate = max(1, round(p95 * effective_safety))

    if n >= 2 * min_samples and not growing:
        confidence: Confidence = "high"
    elif n >= min_samples:
        confidence = "medium"
    else:
        confidence = "low"

    rationale = (
        f"P95={p95}M over {n} sample(s) * safety {effective_safety:.2f} "
        f"= {estimate}M ({'growing' if growing else 'stable'})"
    )
    return MemEstimate(
        estimate,
        confidence,
        n,
        p95,
        slope if n >= 4 else None,
        rationale,
    )
