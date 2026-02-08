import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from astropy.io import fits
from typing import Union, Iterable, Optional
from scipy.signal import periodogram
from dataclasses import dataclass, field

from mirtos.core.types.focal_plane import KID


@dataclass
class Calibration(ABC):
    responsivities: np.ndarray

    @abstractmethod
    def calibrate(self,
                  kids: Iterable[KID],
                  elevations: np.ndarray,
                  inplace: bool = False):
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
                  inplace: bool = False):

        # istruzioni per calibrare le TOD con lo skydip

        tods = np.vstack([k.tod for k in kids])
        ys = np.array([k.pos.y for k in kids])
        resps = np.array([self.responsivities[k.id] for k in kids])

        # dovrebbe essere (236, 1): i fattore di scala per tod
        gains = resps[:, None] * np.exp(- self.tau_atm / np.cos(elevations[None, :] - ys[:, None]))

        if inplace:
            tods /= gains
            for kid, tod, gain in zip(kids, tods, gains):
                # associo ai kid le tod calibrate
                kid.tod, kid.gain = tod, gain

            return None

        return gains, tods / gains

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

            # resps = []
            # for ch, tod in enumerate(tods):
            #     tod = pt["chp_" + str(ch).zfill(3)][mask]
            #     resps.append(np.polyfit(sky_temp, tod, 1)[0])

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

        gains = mean_noise_tot / hfn_feeds

        return gains, mean_noise_tot, hfn_feeds

    def calibrate(self,
                  kids: Iterable[KID],
                  elevations: np.ndarray,
                  inplace: bool = False):

        tods = np.vstack([k.tod for k in kids])
        gains, _, _ = self._compute_gain(tods)

        if inplace:
            tods *= gains[:, None]
            for kid, tod, gain in zip(kids, tods, gains):
                # associo ai kid le tod calibrate
                kid.tod, kid.gain = tod, gain

            return None

        return gains, tods * gains[:, None]


if __name__ == "__main__":
    tau = 0.16
    T_atm = 267
    skydip_003_001_fits = Path("/Volumes/Data/PycharmProjects/mirtos/data/input/20250402-222030-MISTRAL-GAIN_CAL/20250402-222030-MISTRAL-GAIN_CAL_003_001.fits")
    skydip_005_001_fits = Path("/Volumes/Data/PycharmProjects/mirtos/data/input/20250402-000651-MISTRAL-GAIN_CAL/20250402-000651-MISTRAL-GAIN_CAL_005_001.fits")

    sdp_003_001_cal = SkyDipCalibration.from_fits_file(skydip_003_001_fits, T_atm, tau)
    sdp_005_001_cal = SkyDipCalibration.from_fits_file(skydip_005_001_fits, T_atm, tau)

    for sdp_cal, path in zip([sdp_003_001_cal, sdp_005_001_cal], [skydip_003_001_fits, skydip_005_001_fits]):

        output_path = "/Volumes/Data/PycharmProjects/mirtos/tests/data/skydipcalibration/output_skydip_cal_" + path.stem
        np.savez(output_path,
                 resps=sdp_cal.responsivities,
                 taut=tau,
                 Tatm=T_atm)

