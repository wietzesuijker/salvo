from datetime import UTC, datetime, timedelta

import pytest
from salvo.history import JobRecord, estimate_mem, format_suggestion, spec_key


def _record(
    *,
    state: str = "COMPLETED",
    mem_mb: int = 2048,
    max_rss_mb: int | None = 1500,
    ts: datetime | None = None,
    job_id: str = "1",
    key: str = "k",
    cluster: str = "mila",
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        key=key,
        cluster=cluster,
        state=state,
        mem_mb=mem_mb,
        max_rss_mb=max_rss_mb,
        submitted_at=ts or datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
    )


def test_empty_history_returns_none():
    e = estimate_mem([])
    assert e.mem_mb is None
    assert e.confidence == "none"
    assert e.n_samples == 0


def test_insufficient_samples_returns_none():
    e = estimate_mem([_record()])
    assert e.mem_mb is None
    assert e.confidence == "none"
    assert e.n_samples == 1


def test_three_completed_returns_medium_estimate():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(3)]
    e = estimate_mem(records, min_samples=3)
    assert e.mem_mb is not None
    # P95 of three equal values is the value itself; * 1.2 safety -> 4800.
    assert e.mem_mb == 4800
    assert e.confidence == "medium"
    assert e.n_samples == 3


def test_high_confidence_at_double_min_samples():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3)
    assert e.confidence == "high"
    assert e.mem_mb == 4800


def test_p95_picks_upper_tail():
    base = datetime(2026, 5, 1, tzinfo=UTC)
    # Spike placed mid-series so OLS slope stays near zero (isolating P95).
    rss = [1000, 1100, 5000, 1300, 1200]
    records = [
        _record(job_id=str(i), max_rss_mb=v, ts=base + timedelta(hours=i))
        for i, v in enumerate(rss)
    ]
    e = estimate_mem(records, min_samples=3)
    # P95 of these is 5000 (top of distribution); * 1.2 safety = 6000.
    assert e.p95_mb == 5000
    assert e.mem_mb == 6000


def test_oom_record_uses_request_as_lower_bound():
    # OOM at 2G with truncated MaxRSS should be treated as a 2048M observation,
    # not skipped. A real COMPLETED at 1500M should still raise the ceiling.
    base = datetime(2026, 5, 1, tzinfo=UTC)
    records = [
        _record(state="OUT_OF_MEMORY", mem_mb=2048, max_rss_mb=2048, ts=base, job_id="1"),
        _record(
            state="COMPLETED",
            mem_mb=4096,
            max_rss_mb=3000,
            ts=base + timedelta(hours=1),
            job_id="2",
        ),
        _record(
            state="COMPLETED",
            mem_mb=4096,
            max_rss_mb=2900,
            ts=base + timedelta(hours=2),
            job_id="3",
        ),
    ]
    e = estimate_mem(records, min_samples=3)
    assert e.n_samples == 3
    # P95 picks 3000 (the highest observation); 3000 * 1.2 = 3600.
    assert e.p95_mb == 3000
    assert e.mem_mb == 3600


def test_degenerate_max_rss_falls_back_to_req_mem():
    # sacct can miss the peak for short jobs and report MaxRSS as a tiny value
    # even though the job actually used the full allocation. The estimator
    # should treat that as degenerate and use ReqMem instead.
    base = datetime(2026, 5, 1, tzinfo=UTC)
    records = [
        _record(state="OUT_OF_MEMORY", mem_mb=2048, max_rss_mb=1, ts=base, job_id="1"),
        _record(
            state="OUT_OF_MEMORY",
            mem_mb=4096,
            max_rss_mb=0,
            ts=base + timedelta(hours=1),
            job_id="2",
        ),
        # 8G COMPLETED but MaxRSS reads 100M -- clearly bogus.
        _record(
            state="COMPLETED",
            mem_mb=8192,
            max_rss_mb=100,
            ts=base + timedelta(hours=2),
            job_id="3",
        ),
    ]
    e = estimate_mem(records, min_samples=3)
    # Observations should be [2048, 4096, 8192] after falling back on the
    # degenerate COMPLETED row -- P95 = 8192, * 1.2 = 9830.
    assert e.p95_mb == 8192
    assert e.mem_mb == round(8192 * 1.2)


def test_growth_bumps_safety():
    base = datetime(2026, 5, 1, tzinfo=UTC)
    # Monotone increase: 1000, 1200, 1400, 1600, 1800, 2000.
    rss = [1000, 1200, 1400, 1600, 1800, 2000]
    records = [
        _record(job_id=str(i), max_rss_mb=v, ts=base + timedelta(hours=i))
        for i, v in enumerate(rss)
    ]
    e = estimate_mem(records, min_samples=3)
    # Slope is positive; growth bump (1.25x) brings effective safety to 1.5.
    # P95 of these six values is 2000; 2000 * 1.5 = 3000.
    assert e.mem_mb == 3000
    assert e.growth_slope_mb_per_run is not None
    assert e.growth_slope_mb_per_run > 0
    assert e.confidence == "medium"  # growing -> not high


def test_stable_high_confidence_no_growth_bump():
    base = datetime(2026, 5, 1, tzinfo=UTC)
    # Stable around 4000.
    rss = [3900, 4000, 4100, 3950, 4050, 4000, 3900, 4100]
    records = [
        _record(job_id=str(i), max_rss_mb=v, ts=base + timedelta(hours=i))
        for i, v in enumerate(rss)
    ]
    e = estimate_mem(records, min_samples=3)
    assert e.confidence == "high"
    # No growth bump; effective safety stays at 1.2. P95 of these is 4100.
    assert e.mem_mb == round(4100 * 1.2)


def test_window_drops_old_records():
    base = datetime(2026, 5, 1, tzinfo=UTC)
    # Five old huge runs (now stale) + four recent small runs.
    huge = [
        _record(job_id=f"h{i}", max_rss_mb=8000, ts=base + timedelta(hours=i)) for i in range(5)
    ]
    small = [
        _record(job_id=f"s{i}", max_rss_mb=1000, ts=base + timedelta(days=10, hours=i))
        for i in range(4)
    ]
    records = huge + small
    e = estimate_mem(records, min_samples=3, window=4)
    # Window only keeps the four small records.
    assert e.n_samples == 4
    assert e.p95_mb == 1000


def test_dropped_states_are_ignored():
    records = [
        _record(state="PENDING", max_rss_mb=None, job_id="1"),
        _record(state="RUNNING", max_rss_mb=None, job_id="2"),
        _record(state="COMPLETED", max_rss_mb=2000, job_id="3"),
    ]
    e = estimate_mem(records, min_samples=3)
    # Only one learnable record; insufficient.
    assert e.mem_mb is None
    assert e.n_samples == 1


def test_completed_without_max_rss_is_ignored():
    records = [
        _record(state="COMPLETED", max_rss_mb=None, job_id="1"),
        _record(state="COMPLETED", max_rss_mb=None, job_id="2"),
        _record(state="COMPLETED", max_rss_mb=2000, job_id="3"),
    ]
    e = estimate_mem(records, min_samples=3)
    assert e.n_samples == 1


def test_spec_key_is_stable():
    k1 = spec_key("scripts/job.sh", "abc123", ("a", "b"))
    k2 = spec_key("scripts/job.sh", "abc123", ("a", "b"))
    assert k1 == k2


def test_spec_key_changes_with_commit():
    a = spec_key("scripts/job.sh", "abc123")
    b = spec_key("scripts/job.sh", "def456")
    assert a != b


def test_spec_key_changes_with_args_order():
    # Order matters: different arg sequence -> different job.
    a = spec_key("scripts/job.sh", "abc", ("--lr", "0.01"))
    b = spec_key("scripts/job.sh", "abc", ("0.01", "--lr"))
    assert a != b


def test_rationale_is_human_readable():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3)
    assert "P95" in e.rationale
    assert "sample" in e.rationale


@pytest.mark.parametrize("safety", [1.0, 1.5, 2.0])
def test_safety_factor_scales_estimate(safety):
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3, safety=safety)
    assert e.mem_mb == round(4000 * safety)


def test_suggested_floor_matches_mem_mb_when_sufficient():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3)
    assert e.suggested_floor_mb == e.mem_mb
    assert e.suggested_floor_mb == 4800


def test_current_ask_round_trips_through_estimate():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3, current_ask_mb=32768)
    assert e.current_ask_mb == 32768


def test_format_suggestion_insufficient_samples_returns_none():
    e = estimate_mem([_record()], min_samples=3, current_ask_mb=32768)
    assert format_suggestion(e) is None


def test_format_suggestion_exactly_right_ask_returns_none():
    # Six stable runs around 4000M; estimator -> 4800M floor; ask 5000M is well
    # under the 1.5x trigger so no suggestion is emitted.
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    e = estimate_mem(records, min_samples=3, current_ask_mb=5000)
    assert format_suggestion(e) is None


def test_format_suggestion_over_ask_emits_suggestion():
    records = [_record(job_id=str(i), max_rss_mb=4000) for i in range(6)]
    # Floor is 4800M; asking 16384M (16G) is ~3.4x over -> suggestion fires.
    e = estimate_mem(records, min_samples=3, current_ask_mb=16384)
    msg = format_suggestion(e)
    assert msg is not None
    assert "SBATCH_MEM=" in msg
    assert "4800M" in msg or "4800" in msg
    assert "16G" in msg
    assert "3.4x" in msg


def test_format_suggestion_growing_workload_returns_none():
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rss = [1000, 1200, 1400, 1600, 1800, 2000]
    records = [
        _record(job_id=str(i), max_rss_mb=v, ts=base + timedelta(hours=i))
        for i, v in enumerate(rss)
    ]
    # Growing series with a clearly over-allocated 16G ask. Confidence drops
    # to "medium" so the suggestion is suppressed.
    e = estimate_mem(records, min_samples=3, current_ask_mb=16384)
    assert e.confidence == "medium"
    assert format_suggestion(e) is None


def test_format_suggestion_uses_gibibyte_format_on_round_floors():
    # 8192M observation -> floor = round(8192 * 1.2) = 9830M, not a round G;
    # but ask formats as 32G (32768M). Mostly verifying the ask side renders G.
    records = [_record(job_id=str(i), max_rss_mb=8192) for i in range(6)]
    e = estimate_mem(records, min_samples=3, current_ask_mb=32768)
    msg = format_suggestion(e)
    assert msg is not None
    assert "32G" in msg
