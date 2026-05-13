"""Partition picker: walltime + GPU + mem fit, with cluster-specific dispatch rules."""

from __future__ import annotations

from salvo.errors import NoPartitionError
from salvo.job.spec import JobSpec
from salvo.topology.schema import Cluster, Partition


def _partition_fits(p: Partition, *, gpus: int, time_min: int, mem_mb: int) -> bool:
    if time_min > p.max_walltime_hours * 60:
        return False
    if gpus > p.max_gpus_per_job:
        return False
    return not (p.max_mem_gb is not None and mem_mb > p.max_mem_gb * 1024)


def pick_partition(spec: JobSpec, cluster: Cluster, account: str) -> str:
    if spec.partition:
        if not any(p.name == spec.partition for p in cluster.partitions):
            raise NoPartitionError(
                f"explicit partition {spec.partition!r} not in cluster {cluster.id}"
            )
        return spec.partition

    gpus, time_min, mem_mb = spec.gpus, spec.time_min(), spec.mem_mb()
    fitting = [
        p
        for p in cluster.partitions
        if _partition_fits(p, gpus=gpus, time_min=time_min, mem_mb=mem_mb)
    ]
    if not fitting:
        raise NoPartitionError(
            f"no partition on {cluster.id} fits gpus={gpus}, time_min={time_min}, mem_mb={mem_mb}"
        )
    # Prefer GPU partition iff job uses GPUs; among ties prefer shortest walltime ceiling
    fitting.sort(
        key=lambda p: (
            p.max_gpus_per_job == 0 if gpus > 0 else p.max_gpus_per_job > 0,
            p.max_walltime_hours,
        )
    )
    return fitting[0].name
