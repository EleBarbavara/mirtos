from enum import Enum

import yaml
import dacite
from pathlib import Path
from astropy import units as u
from dataclasses import dataclass
from typing import Optional, Union, Any, Iterable

from mirtos.core.types.filters import FilteringConfig, MaskWithoutRadiusMode


@dataclass
class PathsConfig:
    """Paths to expected and calibration files used by the pipeline."""
    instrumentation: Path
    tods: Path
    resp: Union[Path, None]


@dataclass(frozen=True)
class DetectorValidityConfig:
    upper_threshold: float = 7
    lower_threshold: float = 7


class CalibrationType(Enum):
    SKYDIP = 'SKYDIP'
    HF = 'HF'
    NONE = 'NONE'


@dataclass(frozen=True)
class CalibrationConfig:
    tau: float
    T_atm: float
    type: CalibrationType
    path: Path


class MapMakingProjection(Enum):
    SIN = 'SIN'
    GNOM = 'GNOM'


class MapMakingFrame(Enum):
    AZEL = 'AZEL'
    RADEC = 'RADEC'


@dataclass(frozen=True)
class MapMakingConfig:
    """Configuration of the map making """
    pixel_size: u.Quantity
    npix: list[int]


@dataclass
class ScanContext:
    frame: MapMakingFrame
    projection: MapMakingProjection
    beammap_filename: Path
    detector_validity: DetectorValidityConfig
    flag_track: bool = False
    ra_center: float = 0.0
    dec_center: float = 0.0
    angle_offset: float = 0.0


@dataclass(frozen=True)
class PlotMapsConfig:
    """Flags that control which maps are plotted."""
    plot_filt: bool
    plot_unfilt: bool
    plot_counts: bool
    plot_std: bool


@dataclass
class Config:
    """Top-level configuration corresponding to config.yaml."""
    name_target: str
    date_obs: str
    telescope: str
    num_ch_map: Union[str, int]
    paths: PathsConfig
    map_making: MapMakingConfig
    filtering: FilteringConfig
    unfilt_map: str
    plot_maps: PlotMapsConfig
    save_map: bool
    save_single_pixel_maps: bool
    scan: ScanContext
    calibration: CalibrationConfig


def load_config(path: Path) -> Config:
    """Load a YAML configuration file into a typed Config object."""
    data = yaml.safe_load(path.read_text())

    # intermediari per convertire tutti i tipi di input in tipi di output
    # EX: se come dato din input nello yaml gli passo una stringa (radius: "5 arcmin")
    # ma nella dataclass FilteringConfig gli dico che mi aspetto che radius sia di tipo un u.Quantity,
    # raccordo queste due informazioni sui tipi con i type_hooks che fungono da intermediari per convertire
    # tutti i tipi di input nei corretti tipi di output
    type_hooks = {
        MapMakingProjection: MapMakingProjection,
        MapMakingFrame: MapMakingFrame,
        u.Quantity: u.Quantity,
        MaskWithoutRadiusMode: lambda s: MaskWithoutRadiusMode[s.upper()],
        # se non passo nulla a path nel file config, gli viene assegnato None.
        # None deve essere prima convertito in stringa (str(None)) e poi messo in upper case
        # cosi da essere riconosciuto nella classe enum CalibrationType
        CalibrationType: lambda s: CalibrationType[str(s).upper()],
        Path: lambda p: Path(__file__).parents[4] / p if isinstance(p, str) else p,
    }

    return dacite.from_dict(Config, data, config=dacite.Config(type_hooks=type_hooks))


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / 'config.yaml'
    config = load_config(config_path.expanduser().resolve())
    # print(config)
    # print(config.filtering.mask_without_radius)
    # print(config.calibration)
    print(config.scan)
