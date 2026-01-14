from dataclasses import dataclass
from typing import Optional, Union

import yaml


@dataclass
class PathsConfig:
    """Paths to input and calibration files used by the pipeline."""
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


@dataclass
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


def _load_paths(data: dict) -> PathsConfig:
    return PathsConfig(
        tods=data["tods"],
        resp=data.get("resp", False),
        offset_det=data.get("offset_det"),
        skydip=data.get("skydip", False),
        tau=data.get("tau", 0.0),
        T_atm=data.get("T_atm", 0.0),
    )


def _load_binner(data: dict) -> BinnerConfig:
    return BinnerConfig(
        projection=data["projection"],
        frame=data["frame"],
    )


def _load_filtering(data: dict) -> FilteringConfig:
    return FilteringConfig(
        radius=data.get("radius", False),
        baseline_rem=data["baseline_rem"],
        use_detrend_tods=data["use_detrend_tods"],
        gen_cm=data["gen_cm"],
        plot_cm=data["plot_cm"],
        cust_cm=data["cust_cm"],
        plot_corr_matrix=data["plot_corr_matrix"],
        plt_cust_cm=data["plt_cust_cm"],
    )


def _load_plot_maps(data: dict) -> PlotMapsConfig:
    return PlotMapsConfig(
        plot_filt=data["plot_filt"],
        plot_unfilt=data["plot_unfilt"],
        plot_counts=data["plot_counts"],
        plot_std=data["plot_std"],
    )


def load_config(path: str) -> Config:
    """Load a YAML configuration file into a typed Config object."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    num_ch_map_raw = raw.get("num_ch_map", "all")
    if isinstance(num_ch_map_raw, str) and num_ch_map_raw.lower() == "all":
        num_ch_map: Union[str, int] = "all"
    else:
        num_ch_map = int(num_ch_map_raw)

    paths_cfg = _load_paths(raw["paths"])
    binner_cfg = _load_binner(raw["binner"])
    filtering_cfg = _load_filtering(raw["filtering"])
    plot_maps_cfg = _load_plot_maps(raw["plot_maps"])

    return Config(
        name_target=raw["name_target"],
        date_obs=str(raw["date_obs"]),
        telescope=raw["telescope"],
        pixel_size=float(raw["pixel_size"]),
        num_ch_map=num_ch_map,
        paths=paths_cfg,
        flag_track=raw.get("flag_track", True),
        binner=binner_cfg,
        filtering=filtering_cfg,
        unfilt_map=raw["unfilt_map"],
        plot_maps=plot_maps_cfg,
        save_map=raw.get("save_map", False),
        save_single_pixel_maps=raw.get("save_single_pixel_maps", False),
    )