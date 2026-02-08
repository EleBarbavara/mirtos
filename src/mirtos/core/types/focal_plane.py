import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from mirtos.core.types.config import DetectorValidityConfig


@dataclass
class Position:
    x: float
    y: float


@dataclass
class Detector(ABC):
    id: int
    pos: Position
    quality_factor: float
    electrical_responsivity: float
    optical_responsivity: float
    gain: float  # this is the gain that should be estimated with the skydip
    saturation_down: float  # min phase before saturating or going non linear (TBD)
    saturation_up: float  # max phase before saturating or going non linear
    tod: np.ndarray
    validity: DetectorValidityConfig # TODO: parlare di init=True
    # se init fosse True, l'attributo verrebbe creato quando istanzio la classe
    # ma questo genererebbe un problema quando la classe KID eredita Detector
    # perche' all'inizio della classe KID ho attributi obbligatori
    mask: np.ndarray = field(init=False, default_factory=lambda: np.array([]))  # opzionale

    def __post_init__(self):
        # se non e' stato passata alcuna maschera, la creo con tutti i valori False
        if not self.mask.size:
            self.mask = np.zeros_like(self.tod, dtype=bool)

    # attributo astratto che ci dice se la tod e' valida o meno
    @property
    @abstractmethod
    def is_valid(self):
        ...

    @property
    def calibrated_tod(self):
        return self.tod / self.gain


@dataclass
class KID(Detector):
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

        return (data > low) & (data < high).all()


@dataclass
class TES(Detector):

    @property
    def is_valid(self):
        return True


@dataclass
class FocalPlane:
    detectors: list[Detector]

    # TODO: metodo pixel offset


if __name__ == "__main__":
    kid = KID(id=1,
              pos=Position(x=0, y=0),
              quality_factor=1,
              electrical_responsivity=1,
              optical_responsivity=1,
              gain=1,
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
