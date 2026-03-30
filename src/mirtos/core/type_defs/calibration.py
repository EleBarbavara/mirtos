from enum import Enum
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field


class CalibrationType(Enum):
    SKYDIP = 'SKYDIP'
    HF = 'HF'
    NONE = 'NONE'


@dataclass(frozen=True)
class SkyDipCalibrationParams:
    pass


@dataclass
class NoneCalibrationParams:
    pass


@dataclass(frozen=True)
class HFCalibrationParams:
    hf_min_freq: float = 60.0


CalibrationTypeParams = Union[SkyDipCalibrationParams, HFCalibrationParams, NoneCalibrationParams]


@dataclass
class CalibrationConfig:
    tau: float
    T_atm: float
    method: dict
    type: CalibrationType = field(init=False)
    caldir: Union[Path, None] = None

    def __post_init__(self):
        self.type = CalibrationType[self.method["kind"].upper()]
        del self.method["kind"]
