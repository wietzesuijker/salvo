from salvo._version import __version__
from salvo.decorators import cluster
from salvo.job.spec import JobSpec, PythonEntrypoint
from salvo.job.submit import submit

__all__ = ["JobSpec", "PythonEntrypoint", "__version__", "cluster", "submit"]
