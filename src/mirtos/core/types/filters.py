from enum import Enum
from astropy import units as u
from dataclasses import dataclass
from typing import Optional, Any


class MaskWithoutRadiusMode(Enum):
    CUTTED = 'cutted'
    SIGMA = 'sigma'
    NONE = 'none'


@dataclass(frozen=True)
class MaskWithoutRadius:
    mode: MaskWithoutRadiusMode
    offset: float
    sigma: float
    maxiters: int


@dataclass(frozen=True)
class Step:
    """
    dataclasse che contiene le informazioni che stanno sotto le chiavi common, ... di config.yaml
    """
    op: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FilteringSteps:
    common: list[Step]
    mask_with_radius: list[Step]
    mask_without_radius: list[Step]


@dataclass(frozen=True)
class FilteringDebug:
    plot_cm: bool
    plot_corr_matrix: bool


@dataclass(frozen=True)
class FilteringConfig:
    """Configuration for TOD filtering and baseline removal."""
    radius: Optional[u.Quantity]  # in arcsec, or False if not used
    steps: FilteringSteps  # lista di filtri da applicare
    debug: FilteringDebug
    mask_without_radius: MaskWithoutRadius
