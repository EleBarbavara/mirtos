from enum import Enum

import yaml
import dacite
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Any


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


class LinearDetrendMode(Enum):
    CUTTED = 'cutted'
    SIGMA = 'sigma'
    NONE = 'none'

@dataclass(frozen=True)
class Step:
    """
    dataclasse che contiene le informazioni che stanno sotto la chiave steps di config.yaml
    """
    op: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FilteringDebug:
    plot_cm: bool
    plot_corr_matrix: bool


@dataclass(frozen=True)
class FilteringConfig:
    """Configuration for TOD filtering and baseline removal."""
    radius: Union[float, bool]  # in arcsec, or False if not used
    steps: list[Step]  # lista di filtri da applicare
    debug: FilteringDebug

    def __post_init__(self):
        for step in self.steps:
            if step.op == "linear_detrend":
                step.params["mode"] = LinearDetrendMode[step.params["mode"].upper()]


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

    type_hooks = {
        BinnerProjection: BinnerProjection,
        BinnerFrame: BinnerFrame
    }

    return dacite.from_dict(Config, data, config=dacite.Config(type_hooks=type_hooks))


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / 'config.yaml'
    config = load_config(config_path.expanduser().resolve())
    print(config)
    print(config.filtering.steps)
