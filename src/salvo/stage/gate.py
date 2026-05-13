from __future__ import annotations

from salvo.errors import DataNotStagedError
from salvo.job.spec import JobSpec
from salvo.manifest.store import Manifest


def assert_data_available(
    spec: JobSpec,
    *,
    cluster_id: str,
    manifest: Manifest,
    allow_missing: bool = False,
) -> None:
    if not spec.data_needs:
        return
    on_cluster = set(manifest.locations_on(cluster_id))
    missing = set(spec.data_needs) - on_cluster
    if not missing:
        return
    if allow_missing:
        return
    sources = manifest.which_clusters_have(missing)
    raise DataNotStagedError(cluster=cluster_id, missing=missing, sources=sources)
