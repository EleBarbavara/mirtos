import numpy as np
from abc import ABC
from pathlib import Path
from astropy.io import fits
from typing import Union, Iterable
from scipy.signal import periodogram
from dataclasses import dataclass, field

from mirtos.core.types.focal_plane import KID


@dataclass
class Calibration(ABC):
    responsivities: np.ndarray

    def calibrate(self, kids: Union[KID, Iterable[KID]]):
        ...


@dataclass
class SkyDipCalibration(Calibration):

    tau_atm: float
    T_atm: float
    mask: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    airmass: np.ndarray = field(default_factory=lambda: np.array([]))

    def calibrate(self, kids: Union[KID, Iterable[KID]]):
        """
        gli possiamo passare un KID o un insieme di KID
        """
        # istruzioni per calibrare le TOD con lo skydip

        # se e' un KID
        if isinstance(kids, KID):
            denom = self.responsivities[kids.id] * np.exp(- self.tau_atm / np.cos(self.z - kids.pos.y))

            # array TOD calibrato
            return kids.tod / denom

        # altrimenti estriamo le info di tutti gli oggetti KID
        tods = np.vstack([k.tod for k in kids])
        ys = np.array([k.pos.y for k in kids])
        resps = np.array([self.responsivities[k.id] for k in kids])

        denom = resps[:, None] * np.exp(- self.tau_atm / np.cos(self.z[None, :] - ys[:, None]))

        # matrice di TOD calibrate
        return tods / denom

    @classmethod
    def from_fits_file(cls, subscan_filename: Path, T_atm: float, tau_atm: float):
        with fits.open(subscan_filename) as hdul:
            dt = hdul["DATA TABLE"].data
            pt = hdul["PH TABLE"].data

            tot_channels = len(pt[0])
            mask = dt["flag_track"].astype(bool)

            z = np.pi / 2 - dt["el"][mask]
            airmass = 1 / np.cos(z)
            sky_temp = T_atm * (1 - np.exp(-tau_atm * airmass))

            resps = []
            for ch in range(tot_channels):
                tod = pt["chp_" + str(ch).zfill(3)][mask]
                pars = np.polyfit(sky_temp, tod, 1)
                resps.append(pars[0])

        return cls(
            responsivities=np.array(resps),
            tau_atm=tau_atm,
            T_atm=T_atm,
            mask=mask,
            z=z,
            airmass=airmass)


@dataclass
class NoiseCalibration(Calibration):

    sample_freq: float
    hf_min_freq: float = 60

    def _compute_gain(self, tods: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:

        freqs, ps = periodogram(tods, fs=self.sample_freq, scaling="density")

        mask = freqs > self.hf_min_freq
        hfn_feeds = np.sqrt(ps[..., mask]).mean()
        mean_noise_tot = hfn_feeds.mean()

        gains = mean_noise_tot / hfn_feeds

        return gains, mean_noise_tot, hfn_feeds

    def calibrate(self, kids: Union[KID, Iterable[KID]]):

        if isinstance(kids, KID):
            raise NotImplementedError("Come calibro una singola tod quando si usa noise calibration?")

        tods = np.vstack([k.tod for k in kids])
        gains, _, _ = self._compute_gain(tods)

        return tods * gains[:, None]
