"""Append-only events.jsonl emitter. Event taxonomy is stable post-1.0 (spec §17.1).

POSIX-only: uses ``fcntl.flock`` for cross-process serialization of writes,
which matters when epilog + parent + child jobs share the same artifact dir
(and especially on NFS-backed scratch where ``O_APPEND`` atomicity is not
guaranteed for writes above ``PIPE_BUF``). salvo targets Linux/macOS only.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from salvo.errors import EventSchemaError

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
            raise EventSchemaError(f"unknown event {event!r}; add to KNOWN_EVENTS first")
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        with self.path.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
