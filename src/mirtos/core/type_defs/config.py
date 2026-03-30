import sys
import yaml
import dacite
from pathlib import Path
from datetime import datetime
from astropy import units as u
from dataclasses import dataclass, field
from typing import Optional, Union

from mirtos.core.type_defs.beam_map import BeamMap
from mirtos.core.type_defs.calibration import CalibrationConfig
from mirtos.core.type_defs.filters import FilteringConfig, MaskWithoutRadiusMode
from mirtos.core.type_defs.mapmaking import MapMakingProjection, MapMakingFrame, MapMakingConfig


@dataclass
class PathsConfig:
    """Paths to expected and calibration files used by the pipeline."""
    instrumentation: Path
    datadir: Path
    output: Path
    resp: Union[Path, None]

    scan_x_dir: Optional[Path] = field(init=False, default=None)  # RA oppure Az
    scan_y_dir: Optional[Path] = field(init=False, default=None)  # Dec oppure Alt
    calibration_dir: Optional[Path] = field(init=False, default=None)
    data_dirs: list[Path] = field(init=False, default_factory=list)

    @staticmethod
    def _iter_subdirs(datadir: Path) -> list[Path]:
        return sorted([path for path in datadir.iterdir() if path.is_dir()])

    @staticmethod
    def _contains_fits(datadir: Path) -> bool:
        return any(path.is_file() and path.suffix.lower() == ".fits" for path in datadir.iterdir())

    @staticmethod
    def _has_token(name: str, *tokens: str) -> bool:
        upper_name = name.upper()
        for token in tokens:
            upper_token = token.upper()
            if upper_name.endswith(upper_token) or f"_{upper_token}" in upper_name or f"-{upper_token}" in upper_name:
                return True
        return False

    @classmethod
    def _is_calibration_dir(cls, path: Path) -> bool:
        return cls._has_token(path.name, "CAL", "GAIN", "SKYDIP")

    @classmethod
    def _pick_scan_dir(cls, candidates: list[Path], *tokens: str) -> Optional[Path]:
        for path in candidates:
            if cls._has_token(path.name, *tokens):
                return path
        return None

    def resolve_dataset_dirs(self):
        if self._contains_fits(self.datadir):
            self.data_dirs = [self.datadir]
            self.scan_x_dir = self.datadir
            self.scan_y_dir = None
            self.calibration_dir = None
            self.output /= self.datadir.name
            return

        subdirs = self._iter_subdirs(self.datadir)
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories or FITS files found in dataset directory: {self.datadir}")

        calibration_dirs = [path for path in subdirs if self._is_calibration_dir(path)]
        self.calibration_dir = calibration_dirs[0] if calibration_dirs else None

        self.data_dirs = [path for path in subdirs if path not in calibration_dirs]
        if not self.data_dirs:
            raise FileNotFoundError(f"No science subdirectories found in dataset directory: {self.datadir}")

        self.scan_x_dir = self._pick_scan_dir(self.data_dirs, "RA", "AZ")
        remaining_dirs = [path for path in self.data_dirs if path != self.scan_x_dir]

        self.scan_y_dir = self._pick_scan_dir(remaining_dirs, "DEC", "EL", "ALT")

        if self.scan_x_dir is None:
            self.scan_x_dir = self.data_dirs[0]
            remaining_dirs = [path for path in self.data_dirs if path != self.scan_x_dir]

        if self.scan_y_dir is None and remaining_dirs:
            self.scan_y_dir = remaining_dirs[0]

        self.output /= self.datadir.name


@dataclass(frozen=True)
class DetectorValidityConfig:
    upper_threshold: float = 7
    lower_threshold: float = 7


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
    beammap: Optional[BeamMap] = None


@dataclass
class Config:
    """Top-level configuration"""
    name_target: str
    date_obs: datetime
    telescope: str
    num_ch_map: Union[str, int]
    paths: PathsConfig
    map_making: MapMakingConfig
    filtering: FilteringConfig
    scan: ScanContext
    calibration: CalibrationConfig


def load_config(path: Path) -> Config:
    """Load a YAML configuration file into a typed Config object."""
    data = yaml.safe_load(path.read_text())

    # intermediari per convertire tutti i tipi di input in tipi di output
    # EX: se come dato in input nello yaml gli passo una stringa (radius: "5 arcmin")
    # ma nella dataclass FilteringConfig gli dico che mi aspetto che radius sia di tipo un u.Quantity,
    # raccordo queste due informazioni sui tipi con i type_hooks che fungono da intermediari per convertire
    # tutti i tipi di input nei corretti tipi di output
    type_hooks = {
        MapMakingProjection: MapMakingProjection,
        MapMakingFrame: MapMakingFrame,
        u.Quantity: u.Quantity,
        datetime: lambda s: datetime.strptime(s, "%Y%m%d"),
        MaskWithoutRadiusMode: lambda s: MaskWithoutRadiusMode[s.upper()],
        # se non passo nulla a path nel file config, gli viene assegnato None.
        # None deve essere prima convertito in stringa (str(None)) e poi messo in upper case
        # cosi da essere riconosciuto nella classe enum CalibrationType
        # CalibrationType: lambda s: CalibrationType[str(s).upper()],
        Path: lambda p: Path(__file__).parents[4] / p if isinstance(p, str) else p,
    }
    conf = dacite.from_dict(Config, data, config=dacite.Config(type_hooks=type_hooks))
    conf.paths.resolve_dataset_dirs()
    conf.paths.output /= conf.name_target.lower()
    conf.paths.output.mkdir(parents=True, exist_ok=True)

    return conf


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / sys.argv[1]

    if not config_path.exists() or config_path.suffix != ".yaml":
        raise ValueError(f"Config file `{config_path}` does not exist or is not a YAML file")

    config = load_config(config_path.expanduser().resolve())
    print(config.paths.output)
    print(config.paths.datadir)
    print(config.paths.scan_x_dir)
    print(config.paths.scan_y_dir)
    print(config.paths.calibration_dir)
