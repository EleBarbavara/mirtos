import numpy as np
from abc import ABC
from pathlib import Path
from astropy.io import fits
from dataclasses import dataclass, field

from mirtos.core.types.focal_plane import KID


@dataclass
class Calibration(ABC):

    responsivities: np.ndarray

    def calibrate(self, kid: KID):

        ...

class SkyDipCalibration(Calibration):

    tau_atm: float
    T_atm: float
    mask: np.ndarray = field(default_factory=np.array([]))
    z: np.ndarray = field(default_factory=np.array([]))
    airmass: np.ndarray = field(default_factory=np.array([]))

    @property
    def sky_temperature(self):
        return self.T_atm * (1 - np.exp(-self.tau_atm * self.airmass))


    def calibrate(self, kid: KID):
        # istruzioni per calibrare le TOD con lo skydip

        # accediamo alla responsivita' dei soli KID validi
        denom = self.responsivities[kid.id] * np.exp(- self.tau_atm / np.cos(self.z - kid.pos.y))

        return kid.tod / denom

    @classmethod
    def from_fits_file(cls, subscan_filename: Path, T_atm: float, tau_atm: float):

        with fits.open(subscan_filename) as hdul:
            dt = hdul["DATA TABLE"].data
            pt = hdul["PH TABLE"].data

            tot_channels = len(pt[0])
            mask = dt["flag_track"].astype(bool)

            z = np.pi/2 - dt["el"][mask]
            airmass = 1 / np.cos(z)
            sky_temp = T_atm * (1 - np.exp(-tau_atm * airmass))

            resps = []
            for ch in range(tot_channels):

                tod = pt["chp_" + str(ch).zfill(3)][mask]
                pars = np.polyfit(sky_temp, tod, 1)
                resps.append(pars[0])

        return cls(np.array(resps),
                   tau_atm,
                   T_atm,
                   mask,
                   z,
                   airmass)



class GainCalibration(Calibration):
    ...
