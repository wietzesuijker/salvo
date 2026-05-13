"""Append-only events.jsonl emitter. Event taxonomy is stable post-1.0 (spec §17.1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KNOWN_EVENTS: frozenset[str] = frozenset(
    {
        "submit.attempt",
        "submit.success",
        "submit.error",
        "gate.data_missing",
        "dispatch.account_picked",
        "dispatch.partition_picked",
        "hop.oom",
        "hop.preempt",
        "hop.max_exceeded",
        "oom.detected",
        "cancel",
        "done",
    }
)


class EventEmitter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        if event not in KNOWN_EVENTS:
            raise ValueError(f"unknown event {event!r}; add to KNOWN_EVENTS first")
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str, sort_keys=True) + "\n")
