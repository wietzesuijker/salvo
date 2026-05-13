"""Exception hierarchy. All public exceptions inherit from SalvoError."""

from __future__ import annotations


class SalvoError(Exception):
    """Base class for salvo exceptions."""


class ClusterYAMLError(SalvoError):
    """Cluster YAML failed to load or validate."""


class DispatchError(SalvoError):
    """Dispatch (account/partition picking) failed."""


class NoAccountError(DispatchError):
    pass


class NoPartitionError(DispatchError):
    pass


class DataNotStagedError(SalvoError):
    def __init__(
        self,
        cluster: str,
        missing: set[str],
        sources: dict[str, list[str]],
        globus_hint: str | None = None,
    ) -> None:
        self.cluster = cluster
        self.missing = missing
        self.sources = sources
        self.globus_hint = globus_hint
        missing_list = ", ".join(sorted(missing))
        source_hint = "; ".join(
            f"{name}: on {','.join(clusters)}" for name, clusters in sorted(sources.items())
        )
        msg = f"Data not staged on {cluster}: {missing_list}. Sources: {source_hint}"
        if globus_hint:
            msg += f". Hint: {globus_hint}"
        super().__init__(msg)


class MaxHopsExceededError(SalvoError):
    pass
