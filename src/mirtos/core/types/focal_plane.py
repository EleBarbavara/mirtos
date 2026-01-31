from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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


@dataclass
class KID(Detector):
    ch: int  # y
    resonance_freq_hz: float
    sweep_amplitude: np.ndarray = field(default_factory=np.array([]))
    sweep_phase: np.ndarray = field(default_factory=np.array([]))


@dataclass
class TES(Detector):
    ...


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
              sweep_amplitude=1,
              sweep_phase=1,
              tod=np.array([]))

    FocalPlane = FocalPlane([kid])
    print(FocalPlane)
