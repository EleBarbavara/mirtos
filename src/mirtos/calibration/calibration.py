import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from astropy.io import fits
from typing import Union, Iterable, Optional
from scipy.signal import periodogram
from dataclasses import dataclass, field

from mirtos.core.type_defs.focal_plane import KID


@dataclass
class Calibration(ABC):
    responsivities: np.ndarray

    @abstractmethod
    def calibrate(self,
                  kids: Iterable[KID],
                  elevations: np.ndarray,
                  mask: np.ndarray):
        ...


@dataclass
class SkyDipCalibration(Calibration):

    tau_atm: float
    T_atm: float
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    airmass: np.ndarray = field(default_factory=lambda: np.array([]))

    def calibrate(self,
                  kids: Iterable[KID],
                  elevations: np.ndarray,
                  mask: np.ndarray):

        # istruzioni per calibrare le TOD con lo skydip

        # tods = np.vstack([k.tod[mask] for k in kids])
        tods = np.vstack([k.tod for k in kids])
        ys = np.array([k.pos.y for k in kids])
        resps = np.array([self.responsivities[k.id] for k in kids])

        gains = resps[:, None] * np.exp(- self.tau_atm / np.cos(elevations[mask][None, :] - ys[:, None]))

        # salviamo solamente il gain
        for kid, gain in zip(kids, gains):
            kid.gain = gain

        tods[:, mask] /= gains

        return gains, tods

    @classmethod
    def from_fits_file(cls,
                       subscan_filename: Path,
                       T_atm: float,
                       tau_atm: float):

        with fits.open(subscan_filename) as hdul:
            dt = hdul["DATA TABLE"].data
            pt = hdul["PH TABLE"].data

            tot_channels = len(pt[0])

            mask = dt["flag_track"].astype(bool)

            z = np.pi / 2 - dt["el"][mask]
            airmass = 1 / np.cos(z)
            sky_temp = T_atm * (1 - np.exp(-tau_atm * airmass))

            tods = np.vstack([
                pt["chp_" + str(ch).zfill(3)][mask]
                for ch in range(tot_channels)])

            # soluzione chiusa per la regressione lineare
            x0 = sky_temp.mean()
            dx = sky_temp - x0

            den = (dx * dx).sum()
            resps = (tods @ dx) / den

        return cls(
            responsivities=np.array(resps),
            tau_atm=tau_atm,
            T_atm=T_atm,
            mask=mask,
            z=z,
            airmass=airmass)


@dataclass
class HFCalibration(Calibration):

    sample_freq: float
    hf_min_freq: float = 60

    def _compute_gain(self, tods: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:

        freqs, ps = periodogram(tods, fs=self.sample_freq, scaling="density")

        mask = freqs > self.hf_min_freq
        hfn_feeds = np.sqrt(ps[..., mask]).mean()
        mean_noise_tot = hfn_feeds.mean()

        gains = hfn_feeds / mean_noise_tot

        return gains, mean_noise_tot, hfn_feeds

    def calibrate(self,
                  kids: Iterable[KID],
                  elevations: np.ndarray,
                  mask: np.ndarray):

        tods = np.vstack([k.tod[mask] for k in kids])
        gains, _, _ = self._compute_gain(tods)

        # salviamo solamente il gain
        for kid, gain in zip(kids, gains):
            kid.gain = gain

        return gains, tods / gains[:, None]


if __name__ == "__main__":
    tau = 0.16
    T_atm = 267

    p = Path("/Volumes/Data/PycharmProjects/mirtos/data/input/20250403-023545/20250403-033037-MISTRAL-GAIN_CAL/20250402-222030-MISTRAL-GAIN_CAL_003_001.fits")
    sdp_003_001_cal = SkyDipCalibration.from_fits_file(p, T_atm, tau)
    print(sdp_003_001_cal)
