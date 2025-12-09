from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.constants import c


@dataclass
class TelescopeConfig:
    """Generic description of a single-dish telescope."""
    name: str
    diameter_m: float
    central_freq_hz: Optional[float] = None
    central_wavelength_m: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    efficiency: float = 1.0
    fov_arcmin: Optional[float] = None

    def __post_init__(self) -> None:
        if self.central_freq_hz is None and self.central_wavelength_m is None:
            raise ValueError(
                "Either 'central_freq_hz' or 'central_wavelength_m' must be provided."
            )

        # Deriva la frequenza se hai passato solo la lunghezza d’onda
        if self.central_freq_hz is None and self.central_wavelength_m is not None:
            self.central_freq_hz = c / self.central_wavelength_m

        # Deriva la lunghezza d’onda se hai passato solo la frequenza
        if self.central_wavelength_m is None and self.central_freq_hz is not None:
            self.central_wavelength_m = c / self.central_freq_hz

    @property
    def beam_fwhm_rad(self) -> float:
        return self.central_wavelength_m / self.diameter_m

    @property
    def beam_solid_angle_sr(self) -> float:
        return np.pi * self.beam_fwhm_rad**2 / (4 * np.log(2.0))

    @property
    def collecting_area_m2(self) -> float:
        return np.pi * (self.diameter_m / 2.0) ** 2

    @property
    def upper_band_edge_hz(self) -> Optional[float]:
        if self.bandwidth_hz is None:
            return None
        return self.central_freq_hz + 0.5 * self.bandwidth_hz

    @property
    def lower_band_edge_hz(self) -> Optional[float]:
        if self.bandwidth_hz is None:
            return None
        return self.central_freq_hz - 0.5 * self.bandwidth_hz


@dataclass
class KIDConfig:
    """
    Configuration for a Kinetic Inductance Detector (KID).

    Parameters
    ----------
    id : str
        Identifier or label of the detector.
    resonance_freq_hz : float
        Resonance frequency of the KID in Hz.
    """
    id: str
    resonance_freq_hz: float