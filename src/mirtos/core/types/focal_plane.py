from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Detector(ABC):

    id: int
    quality_factor: float
    electrical_responsivity: float
    optical_responsivity: float
    gain: float  # this is the gain that should be estimated with the skydip
    saturation_down: float  # min phase before saturating or going non linear (TBD)
    saturation_up: float  # max phase before saturating or going non linear

@dataclass
class KID(Detector):

    ch: int  # y
    resonance_freq_hz: float
    sweep_amplitude: float  ## come metto un np.array o simile? Sarebbe carino memorizzare proprio gli sweep per fittarli eventualmente con altre funzioni che sta scrivendo il laureando di Alepaiella
    sweep_phase: float  ## idem


@dataclass
class TES(Detector):
    pass


@dataclass
class FocalPlane:

    detectors: list[Detector]


    # TODO: metodo pixel offset

if __name__ == "__main__":

    kid = KID(id=1,
              quality_factor=1,
              electrical_responsivity=1,
              optical_responsivity=1,
              gain=1,
              saturation_down=1,
              saturation_up=1,
              ch=1,
              resonance_freq_hz=1,
              sweep_amplitude=1,
              sweep_phase=1)

    FocalPlane = FocalPlane([kid])
