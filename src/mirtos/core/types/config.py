from enum import Enum

import yaml
import dacite
from pathlib import Path
from astropy import units as u
from dataclasses import dataclass
from typing import Optional, Union, Any

from mirtos.core.types.filters import FilteringConfig, MaskWithoutRadiusMode


@dataclass
class PathsConfig:
    """Paths to expected and calibration files used by the pipeline."""
    instrumentation: str
    tods: str
    resp: Union[str, bool]
    offset_det: Optional[str]
    skydip: Union[str, bool]
    tau: float
    T_atm: float


@dataclass(frozen=True)
class DetectorValidityConfig:
    upper_threshold: float = 7
    lower_threshold: float = 7


class BinnerProjection(Enum):
    SIN = 'SIN'
    GNOM = 'GNOM'


class BinnerFrame(Enum):
    AZEL = 'AZEL'
    RADEC = 'RADEC'


@dataclass(frozen=True)
class BinnerConfig:
    """Configuration of the map binning """
    projection: BinnerProjection  # e.g. 'SIN' or 'GNOM'
    frame: BinnerFrame  # e.g. 'AZEL', 'RADEC', 'EQ'



@dataclass
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
    pixel_size: float
    num_ch_map: Union[str, int]
    paths: PathsConfig
    flag_track: bool
    binner: BinnerConfig
    filtering: FilteringConfig
    unfilt_map: str
    plot_maps: PlotMapsConfig
    save_map: bool
    save_single_pixel_maps: bool
    detector_validity: DetectorValidityConfig

    def __post_init__(self):
        base_path = Path(__file__).parents[4]

        for field in (
                "instrumentation",
                "tods",
                "resp",
                "offset_det",
                "skydip"):

            value = getattr(self.paths, field)

            if isinstance(value, str):
                setattr(self.paths, field, (base_path / value).expanduser().resolve())
                assert getattr(self.paths, field).exists(), f"{field} does not exist: {getattr(self.paths, field)}"


def load_config(path: Path) -> Config:
    """Load a YAML configuration file into a typed Config object."""
    data = yaml.safe_load(path.read_text())

    # intermediari per convertire tutti i tipi di input in tipi di output
    # EX: se come dato din input nello yaml gli passo una stringa (radius: "5 arcmin")
    # ma nella dataclass FilteringConfig gli dico che mi aspetto che radius sia di tipo un u.Quantity,
    # raccordo queste due informazioni sui tipi con i type_hooks che fungono da intermediari per convertire
    # tutti i tipi di input nei corretti tipi di output
    type_hooks = {
        BinnerProjection: BinnerProjection,
        BinnerFrame: BinnerFrame,
        u.Quantity: u.Quantity,
        MaskWithoutRadiusMode: lambda s: MaskWithoutRadiusMode[s.upper()]
    }

    return dacite.from_dict(Config, data, config=dacite.Config(type_hooks=type_hooks))


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / 'config.yaml'
    config = load_config(config_path.expanduser().resolve())
    print(config)
    print(config.filtering.mask_without_radius)
