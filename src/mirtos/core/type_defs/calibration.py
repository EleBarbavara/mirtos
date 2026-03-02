from enum import Enum
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field


class CalibrationType(Enum):
    # ogni istanza di una classe enum deve avere un valore univoco, cioe' puo'
    # valere solo uno di questi attributi
    SKYDIP = 'SKYDIP'
    HF = 'HF'
    NONE = 'NONE'


@dataclass(frozen=True)
class SkyDipCalibrationParams:
    # FIXME: qui dovremmo definire resp
    pass


@dataclass
class NoneCalibrationParams:
    pass


@dataclass(frozen=True)
class HFCalibrationParams:
    hf_min_freq: float = 60.0

# Union puo' avere un solo tipo per istanza
CalibrationTypeParams = Union[SkyDipCalibrationParams, HFCalibrationParams, NoneCalibrationParams]


@dataclass
class CalibrationConfig:
    tau: float
    T_atm: float
    method: dict
    type: CalibrationType = field(init=False)
    path: Path = field(init=False)

    def __post_init__(self):
        # converto kind da stringa a tipo CalibrationType
        self.type = CalibrationType[self.method["kind"].upper()]
        # dato che ora ho self.type posso cancellare self.method["kind"]
        # dal dizionario
        del self.method["kind"]
