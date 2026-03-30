import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from mirtos.core.type_defs.config import DetectorValidityConfig


@dataclass
class Position:
    x: float
    y: float

    def __cmp__(self, other):
        return self.x == other.x and self.y == other.y

    @property
    def r(self):
        # ordinamento per raggio (pattern radiali)
        return np.hypot(self.x, self.y)

    @property
    def theta(self):
        # ordinamento polare (pattern angolari)
        return np.arctan2(self.y, self.x)


@dataclass
class Detector(ABC):
    id: int
    pos: Position
    quality_factor: float
    electrical_responsivity: float
    optical_responsivity: float
    gain: np.ndarray  # this is the gain (time-based) that should be estimated with the skydip
    saturation_down: float  # min phase before saturating or going non linear (TBD)
    saturation_up: float  # max phase before saturating or going non linear
    tod: np.ndarray
    validity: DetectorValidityConfig
    # se init fosse True, l'attributo verrebbe creato quando istanzio la classe
    # ma questo genererebbe un problema quando la classe KID eredita Detector
    # perche' all'inizio della classe KID ho attributi obbligatori
    mask: np.ndarray = field(init=False, default_factory=lambda: np.array([]))  # opzionale

    _is_calibrated: bool = field(init=False, default=False)

    def __post_init__(self):
        # se non e' stato passata alcuna maschera, la creo con tutti i valori True
        if not self.mask.size:
            self.mask = np.ones_like(self.tod, dtype=bool)

    @property
    def is_calibrated(self):
        return self._is_calibrated

    # attributo astratto che ci dice se la tod e' valida o meno
    @property
    @abstractmethod
    def is_valid(self):
        ...


@dataclass
class KID(Detector):
    @property
    def is_calibrated(self):
        return self._is_calibrated

    ch: int  # y
    resonance_freq_hz: float
    sweep_amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    sweep_phase: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_valid(self):
        upper_threshold = self.validity.upper_threshold
        lower_threshold = self.validity.lower_threshold

        if not (data := self.tod[self.mask]).size:
            return False

        mean = data.mean()
        std = data.std()

        low = mean - lower_threshold * std
        high = mean + upper_threshold * std

        return ((data > low) & (data < high)).all()

    def apply_calibration_inplace(self, cal_tod: np.ndarray):
        self.tod = cal_tod
        self._is_calibrated = True

    @is_calibrated.setter
    def is_calibrated(self, value):
        self._is_calibrated = value


@dataclass
class TES(Detector):

    @property
    def is_valid(self):
        return True


@dataclass
class FocalPlane:
    detectors: list[Detector]


if __name__ == "__main__":
    kid = KID(id=1,
              pos=Position(x=0, y=0),
              quality_factor=1,
              electrical_responsivity=1,
              optical_responsivity=1,
              gain=np.array([]),
              saturation_down=1,
              saturation_up=1,
              ch=1,
              resonance_freq_hz=1,
              sweep_amplitude=np.array([]),
              sweep_phase=np.array([]),
              tod=np.array([]),
              validity=DetectorValidityConfig())

    fp = FocalPlane([kid])
    print(fp)
