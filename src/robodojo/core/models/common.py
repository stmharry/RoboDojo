"""Shared primitives for typed RoboDojo domain models."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegativeInt = Annotated[int, Field(ge=0)]
Port = Annotated[int, Field(ge=1, le=65535)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataFormat(StrEnum):
    LEROBOT_V30 = "lerobot_v3.0"
    LEROBOT_V21 = "lerobot_v2.1"
    HDF5 = "hdf5"
    DEMO = "demo"
    REAL = "real"


def finite_vector(value: tuple[float, ...], *, length: int, field: str) -> tuple[float, ...]:
    if len(value) != length or not all(math.isfinite(component) for component in value):
        raise ValueError(f"{field} must contain {length} finite values")
    return value
