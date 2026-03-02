import yaml
import dacite
from pathlib import Path
from datetime import datetime
from astropy import units as u
from dataclasses import dataclass, field
from typing import Optional, Union

from mirtos.core.type_defs.beam_map import BeamMap
from mirtos.core.type_defs.calibration import CalibrationConfig, CalibrationType
from mirtos.core.type_defs.filters import FilteringConfig, MaskWithoutRadiusMode
from mirtos.core.type_defs.mapmaking import MapMakingProjection, MapMakingFrame, MapMakingConfig


@dataclass
class PathsConfig:
    """Paths to expected and calibration files used by the pipeline."""
    # FIXME: per ora telescope e' di tipo Path, ma bisognerebbe creare in typedefs due file
    #   telescope.py e receiver.py con dataclasses per gestire i dati contenuti in receiver.yaml e telescope.yaml
    instrumentation: Path
    datadir: Path
    resp: Union[Path, None]

    # init=False vuol dire che a runtime non viene definita. viene definita solo
    # quando viene creato l'oggetto PathsConfig
    ra_dir: Path = field(init=False)
    dec_dir: Path = field(init=False)
    gain_dir: Path = field(init=False)

    def __post_init__(self):
        # glob() restituisce un iteratore, quindi devo usare next() per ottenere il primo elemento
        # in questo caso ci aspettiamo una sola cartella che abbia nel suo nome "RA", una sola che
        # abbia nel suo nome "DEC" e una sola che abbia nel suo nome "CAL"
        # quindi va bene usare next in quanto ritorna il primo elemento di un iteratore che contiene un solo elemento
        # FIXME: i nomi delle cartelle devo avere per forza "RA", DEC" e "cal" (case insensitive) al momento
        self.ra_dir = next(self.datadir.glob("*RA", case_sensitive=False), None)
        self.dec_dir = next(self.datadir.glob("*DEC", case_sensitive=False), None)
        self.gain_dir = next(self.datadir.glob("*CAL", case_sensitive=False), None)


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
    """Top-level configuration corresponding to a1995.yaml."""
    name_target: str
    date_obs: datetime
    # FIXME: per ora telescope e' una stringa, ma bisognerebbe creare in typedefs due file
    #   telescope.py e receiver.py con dataclasses per gestire i dati contenuti in receiver.yaml e telescope.yaml
    telescope: str
    # FIXME: modificare in modo che accetti un intero, -1, un range
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
    # EX: se come dato din input nello yaml gli passo una stringa (radius: "5 arcmin")
    # ma nella dataclass FilteringConfig gli dico che mi aspetto che radius sia di tipo un u.Quantity,
    # raccordo queste due informazioni sui tipi con i type_hooks che fungono da intermediari per convertire
    # tutti i tipi di input nei corretti tipi di output
    type_hooks = {
        MapMakingProjection: MapMakingProjection,
        MapMakingFrame: MapMakingFrame,
        u.Quantity: u.Quantity,
        # s parametro della funzione anonima lambda
        datetime: lambda s: datetime.strptime(s, "%Y%m%d"),
        # MaskWithoutRadiusMode ha gli attributi definiti tutti in maiuscolo per convenzione
        # quindi, se dovessero essre scritti in minuscolo nel file di config, li
        # traformo in maiuscolo
        MaskWithoutRadiusMode: lambda s: MaskWithoutRadiusMode[s.upper()],
        # per ottenere un oggetto di tipo Path, definisco una funzione lambda con parametro p
        # che effettua alcune istruzioni e ritorna un oggetto di tipo Path.
        # Serve in quanto nel file 3c84.yaml ho stringhe e non Path, ma io voglio
        # lavorare con tipi Path.
        # Ovunque ci sia nei codici un tipo Path, viene richiamato questo type_hook qui
        # l'else assegna None se p non e' una stringa
        Path: lambda p: Path(__file__).parents[4] / p if isinstance(p, str) else p,
    }

    # ritorno un oggetto Config a partire dal dizionario costruito dallo yaml
    # Config infatti e' una dataclass che contiene tutti i campi dello yaml.
    # dacite viene usato per traformare un dizionario in una dataclass
    return dacite.from_dict(Config, data, config=dacite.Config(type_hooks=type_hooks))


if __name__ == "__main__":
    config_path = Path(__file__).parents[4] / 'configs' / 'a1995.yaml'
    config = load_config(config_path.expanduser().resolve())
    # print(config)
    # print(config.filtering.mask_without_radius)
    # print(config.calibration)
    print(config.paths)
