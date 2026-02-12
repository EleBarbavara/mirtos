import yaml
import numpy as np
from pathlib import Path
import astropy.units as u
from dataclasses import dataclass

from mirtos.io.yaml import read_yaml

# non vengono modificate a run_time fissato
@dataclass(frozen=True, slots=True)
class TelescopeConfig:

    name: str
    efficiency: float
    diameter: u.Quantity


    @property
    def area(self):
        return np.pi * (self.diameter / 2)**2

    @classmethod
    def from_yaml(cls, filename: Path):
        return from_yaml(cls, filename)


@dataclass(frozen=True, slots=True)
class ReceiverConfig:

    name: str
    efficiency: float
    bandwidth: u.Quantity
    central_freq: u.Quantity
    central_wavelength: u.Quantity
    fov: u.Quantity

    # ReceiverConfig.end_band e non ReceiverConfig.end_band()
    # lo possiamo fare solo se il metodo non accetta parametri in ingresso
    @property
    def end_band(self):
        return self.central_freq + self.band_width

    @classmethod
    def from_yaml(cls, filename: Path):
        return from_yaml(cls, filename)


@dataclass(frozen=True, slots=True)
class InstrumentConfig:

    telescope: TelescopeConfig
    receiver: ReceiverConfig

    @property
    def beam(self):
        return self.receiver.central_freq / self.telescope.diameter

    @property
    def beam_area(self):
        return np.pi * self.beam**2 / (4 * np.log(2))

    @classmethod
    def from_yaml(cls, filename: Path):

        cfg = read_yaml(filename)

        t_path = Path(filename).expanduser().resolve().parent / cfg['telescope']
        r_path = Path(filename).expanduser().resolve().parent / cfg['receiver']

        telescope = TelescopeConfig.from_yaml(t_path)
        receiver = ReceiverConfig.from_yaml(r_path)

        return cls(telescope=telescope, receiver=receiver)


if __name__ == "__main__":

    telescope_path = Path('/configs/instrument/telescope/SRT.yaml')
    receiver_path = Path('/configs/instrument/receiver/mistral.yaml')
    instrument_path = Path('/configs/instrument/mistral_SRT.yaml')

    tele_cfg = TelescopeConfig.from_yaml(telescope_path)
    rec_cfg = ReceiverConfig.from_yaml(receiver_path)
    inst_cfg = InstrumentConfig.from_yaml(instrument_path)
    print(tele_cfg)
    print(rec_cfg)
    print(inst_cfg)