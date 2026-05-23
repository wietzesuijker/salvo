from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from salvo.errors import _run_subprocess
from salvo.job.spec import Hop, JobSpec, JobStatus

# sacct MaxRSS field unit multipliers (to kilobytes).
_MAX_RSS_UNITS = {"K": 1, "M": 1024, "G": 1024 * 1024, "T": 1024 * 1024 * 1024}


def _parse_max_rss_kb(raw: str) -> int | None:
    """Parse sacct MaxRSS (e.g. ``1024K``, ``4096M``, ``2G``) to kilobytes.

    Returns None for empty/whitespace-only input. Unrecognized suffixes also
    return None (defensive: sacct shouldn't emit them, but don't crash status()).
    """
    s = raw.strip()
    if not s:
        return None
    unit = s[-1].upper()
    if unit in _MAX_RSS_UNITS:
        try:
            value = float(s[:-1])
        except ValueError:
            return None
        return int(value * _MAX_RSS_UNITS[unit])
    # No suffix: assume bytes per sacct convention -> kilobytes.
    try:
        return int(s) // 1024
    except ValueError:
        return None


class JobHandle:
    def __init__(self, *, job_id: str, cluster: str, spec: JobSpec, artifact_dir: Path) -> None:
        self.job_id = job_id
        self.cluster = cluster
        self.spec = spec
        self.artifact_dir = artifact_dir

    def status(self) -> JobStatus:
        out = _run_subprocess(
            [
                "sacct",
                "-j",
                self.job_id,
                "-P",
                "-o",
                "JobID,State,ExitCode,Elapsed,MaxRSS",
                "-n",
            ],
            check=False,
            timeout=10,
        )
        parent: list[str] | None = None
        max_rss_kb: int | None = None
        for line in out.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            job_id = parts[0]
            if "." not in job_id:
                parent = parts
            elif job_id.endswith(".batch") and len(parts) >= 5:
                max_rss_kb = _parse_max_rss_kb(parts[4])
        if parent is None:
            return JobStatus(job_id=self.job_id, state="UNKNOWN")
        return JobStatus(
            job_id=parent[0],
            state=parent[1].strip(),
            max_rss_kb=max_rss_kb,
        )

    def cancel(self) -> None:
        _run_subprocess(["scancel", self.job_id], check=False, timeout=10)

    def logs(
        self, stream: Literal["stdout", "stderr"] = "stdout", follow: bool = False
    ) -> Iterator[str]:
        log = self.artifact_dir / f"{stream}.log"
        if not log.exists():
            return iter([])
        if not follow:
            return iter(log.read_text().splitlines())

        def _tail() -> Iterator[str]:
            with log.open() as f:
                while True:
                    line = f.readline()
                    if not line:
                        return
                    yield line.rstrip("\n")

        return _tail()

    def hops(self) -> list[Hop]:
        hops_file = self.artifact_dir / "hops.jsonl"
        if not hops_file.exists():
            return []
        lines = hops_file.read_text().splitlines()
        return [Hop.model_validate_json(line) for line in lines if line.strip()]
