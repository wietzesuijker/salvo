from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetLocation(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    verified_at: datetime
    size_gb: float | None = None
    checksum: str | None = None


class Dataset(BaseModel):
    description: str = ""
    checksum: str | None = None
    locations: dict[str, DatasetLocation] = Field(default_factory=dict)
