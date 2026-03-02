from enum import Enum
import astropy.units as u
from dataclasses import dataclass


class MapMakingProjection(Enum):
    SIN = 'SIN'
    GNOM = 'GNOM'


class MapMakingFrame(Enum):
    AZEL = 'AZEL'
    RADEC = 'RADEC'


@dataclass(frozen=True)
class MapMakingConfig:
    """Configuration of the map-making """
    pixel_size: u.Quantity
    # map's resolution in pixels
    npix: list[int]
