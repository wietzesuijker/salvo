from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from salvo.job.spec import Hop, JobSpec, JobStatus


class JobHandle:
    def __init__(self, *, job_id: str, cluster: str, spec: JobSpec, artifact_dir: Path) -> None:
        self.job_id = job_id
        self.cluster = cluster
        self.spec = spec
        self.artifact_dir = artifact_dir

    def status(self) -> JobStatus:
        out = subprocess.run(
            [
                "sacct",
                "-j",
                self.job_id,
                "-X",
                "-P",
                "-o",
                "JobID,State,ExitCode,Elapsed,MaxRSS",
                "-n",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in out.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 2:
                continue
            return JobStatus(job_id=parts[0], state=parts[1].strip())
        return JobStatus(job_id=self.job_id, state="UNKNOWN")

    def cancel(self) -> None:
        subprocess.run(["scancel", self.job_id], check=False)

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
        import json  # noqa: F401

        lines = hops_file.read_text().splitlines()
        return [Hop.model_validate_json(line) for line in lines if line.strip()]
