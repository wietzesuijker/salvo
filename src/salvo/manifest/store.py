"""TOML-backed dataset manifest with atomic + flocked writes."""

from __future__ import annotations

import fcntl
import os
import tempfile
import tomllib  # py3.11+
import types
from pathlib import Path

_tomli_w: types.ModuleType | None
try:
    import tomli_w

    _tomli_w = tomli_w
except ImportError:
    _tomli_w = None  # falls back to manual TOML emission below

from salvo.manifest.schema import Dataset, DatasetLocation  # noqa: E402


class Manifest:
    def __init__(self, path: Path, datasets: dict[str, Dataset]) -> None:
        self.path = path
        self.datasets = datasets

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls(path, {})
        with path.open("rb") as f:
            raw = tomllib.load(f)
        ds_raw = raw.get("datasets", {})
        datasets: dict[str, Dataset] = {}
        for name, body in ds_raw.items():
            locs = {cid: DatasetLocation(**loc) for cid, loc in body.pop("locations", {}).items()}
            datasets[name] = Dataset(**body, locations=locs)
        return cls(path, datasets)

    def record(self, name: str, *, cluster: str, location: DatasetLocation) -> None:
        ds = self.datasets.setdefault(name, Dataset())
        ds.locations[cluster] = location

    def save(self) -> None:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                body = self._dump()
                fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".tmp.", dir=self.path.parent)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(body)
                    os.replace(tmp, self.path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _dump(self) -> bytes:
        if _tomli_w is None:
            lines = ["schema_version = 1\n"]
            for name, ds in self.datasets.items():
                lines.append(f"\n[datasets.{name}]\n")
                if ds.description:
                    lines.append(f'description = "{ds.description}"\n')
                if ds.checksum:
                    lines.append(f'checksum = "{ds.checksum}"\n')
                for cid, loc in ds.locations.items():
                    lines.append(f"\n[datasets.{name}.locations.{cid}]\n")
                    lines.append(f'path = "{loc.path}"\n')
                    lines.append(f"verified_at = {loc.verified_at.isoformat()!r}\n")
                    if loc.size_gb is not None:
                        lines.append(f"size_gb = {loc.size_gb}\n")
                    if loc.checksum:
                        lines.append(f'checksum = "{loc.checksum}"\n')
            return "".join(lines).encode()
        return bytes(
            _tomli_w.dumps(
                {
                    "schema_version": 1,
                    "datasets": {
                        name: {
                            "description": ds.description,
                            "locations": {
                                cid: {
                                    "path": loc.path,
                                    "verified_at": loc.verified_at,
                                    **({"size_gb": loc.size_gb} if loc.size_gb is not None else {}),
                                    **({"checksum": loc.checksum} if loc.checksum else {}),
                                }
                                for cid, loc in ds.locations.items()
                            },
                        }
                        for name, ds in self.datasets.items()
                    },
                }
            ).encode()
        )

    def locations_on(self, cluster: str) -> list[str]:
        return [name for name, ds in self.datasets.items() if cluster in ds.locations]

    def which_clusters_have(self, names: set[str]) -> dict[str, list[str]]:
        return {n: sorted(self.datasets[n].locations) for n in names if n in self.datasets}
