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


@dataclass
class BinnerConfig:
    """Configuration of the map binning."""
    projection: str  # e.g. 'SIN' or 'GNOM'
    frame: str       # e.g. 'AZEL', 'RADEC', 'EQ'


class StepScope(Enum):
    LOCAL = 1
    GLOBAL = 2


@dataclass(frozen=True)
class Step:
    """
    dataclasse che contiene le informazioni che stanno sotto la chiave steps di config.yaml
    """
    op: str
    scope: StepScope
    params: dict[str, Any]


@dataclass(frozen=True)
class FilteringConfig:
    """Configuration for TOD filtering and baseline removal."""
    radius: Union[float, bool]  # in arcsec, or False if not used
    baseline_rem: str           # e.g. 'cutted'
    use_detrend_tods: bool
    gen_cm: bool
    plot_cm: bool
    cust_cm: bool
    plot_corr_matrix: bool
    plt_cust_cm: bool
    steps: list[Step] # lista di filtri da applicare


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
                setattr(self.paths,field, (base_path / value).expanduser().resolve())
                assert getattr(self.paths, field).exists(), f"{field} does not exist: {getattr(self.paths, field)}"


def load_config(path: Path) -> Config:
    """Load a YAML configuration file into a typed Config object."""
    data = yaml.safe_load(path.read_text())

    dacite_cfg = dacite.Config(
        type_hooks={StepScope: lambda v: (StepScope[v.upper()] if isinstance(v, str) else StepScope(v))}
    )

    return dacite.from_dict(Config, data, config=dacite_cfg)


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / 'config.yaml'
    config = load_config(config_path.expanduser().resolve())
    print(config)
    print(config.filtering.steps)