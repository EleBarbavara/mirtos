from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from astropy import units as u


class MaskWithoutRadiusMode(Enum):
    CUT = "cut"
    SIGMA = "sigma"
    NONE = "none"


@dataclass(frozen=True)
class Step:
    """
    One user-switchable filtering operation.

    The operation name must match a function registered in filtering/filters.py.
    Mask construction is configured outside steps.
    """
    op: str
    params: dict[str, Any]


@dataclass(frozen=True)
class MaskWithoutRadius:
    """Parameters used to build the primary mask when radius is zero."""
    mode: MaskWithoutRadiusMode
    offset: float
    sigma: float
    maxiters: int


@dataclass(frozen=True)
class FilteringDebug:
    plot_cm: bool
    plot_corr_matrix: bool


@dataclass(frozen=True)
class FilteringConfig:
    """Configuration for TOD filtering and baseline removal."""
    radius: Optional[u.Quantity]
    without_radius: MaskWithoutRadius
    steps: list[Step]
    debug: FilteringDebug