import sys
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d

from mirtos.core.type_defs.scan import Scan
from mirtos.core.projections import conv_radec_to_latlon
from mirtos.core.type_defs.config import MapMakingFrame, load_config
from mirtos.plotting.map import plot_map


def _make_bins(npix_x: int, npix_y: int, pixel_size_deg: float,
               center_ra_deg: float, center_dec_deg: float):

    x_min = center_ra_deg - (npix_x // 2) * pixel_size_deg
    x_max = center_ra_deg + (npix_x // 2) * pixel_size_deg
    y_min = center_dec_deg - (npix_y // 2) * pixel_size_deg
    y_max = center_dec_deg + (npix_y // 2) * pixel_size_deg
    x_bins = np.linspace(x_min, x_max, npix_x + 1)
    y_bins = np.linspace(y_min, y_max, npix_y + 1)
    return x_bins, y_bins


def _do_binning(x: np.ndarray, y: np.ndarray, values: np.ndarray, bins, range_):
    data_map, *_ = binned_statistic_2d(
        x, y,
        values=values,
        statistic="mean",
        bins=bins,
        range=range_)

    count_map, *_ = binned_statistic_2d(
        x, y,
        values=values,
        statistic="count",
        bins=bins,
        range=range_)

    return data_map, count_map


class MapMaker(ABC):

    def __init__(self,
                 scans: list[Scan],
                 pixel_size: u.Quantity,
                 npix: list[int]):

        self.scans = scans
        self.pixel_size = pixel_size
        self.npix = npix

        self.frame = scans[0].ctx.frame
        self.projection = scans[0].ctx.projection

        self.ra_center = scans[0].ctx.ra_center
        self.dec_center = scans[0].ctx.dec_center

    @abstractmethod
    def combine_scans(self):
        ...

    @abstractmethod
    def make_map(self):
        ...


class BinnerMapMaker(MapMaker):

    def combine_scans(self):
        ...

    def _make_wcs(self, ctype1: str, ctype2: str, crval1: float, crval2: float,
                  npix_x: int, npix_y: int, pixel_size_deg: float) -> WCS:
        wcs_dict = {
            "CTYPE1": f"{ctype1}{self.projection}",
            "CUNIT1": "deg",
            "CDELT1": -pixel_size_deg,
            "CRPIX1": (npix_x + 1) / 2,
            "CRVAL1": crval1,
            "NAXIS1": npix_x,
            "CTYPE2": f"{ctype2}{self.projection}",
            "CUNIT2": "deg",
            "CDELT2": pixel_size_deg,
            "CRPIX2": (npix_y + 1) / 2,
            "CRVAL2": crval2,
            "NAXIS2": npix_y,
        }
        return WCS(wcs_dict)

    def make_map(self):
        npix_x, npix_y = self.npix
        pixel_size_deg = self.pixel_size.to_value('deg')
        center_ra_deg = np.rad2deg(self.ra_center)
        center_dec_deg = np.rad2deg(self.dec_center)

        scan_ = self.scans[0]
        # appiattiamo tutte le tod (concateno i kid)
        values = scan_.tods.ravel()

        lon, lat = conv_radec_to_latlon(
            scan_.ra,
            scan_.dec,
            self.ra_center,
            self.dec_center,
            self.projection,
            scan_.par_angle,
            scan_.ctx.beammap.beam_map['lon_offset'].to_numpy(),
            scan_.ctx.beammap.beam_map['lat_offset'].to_numpy(),
            self.frame)

        if self.frame == MapMakingFrame.RADEC:
            x_bins, y_bins = _make_bins(npix_x, npix_y, pixel_size_deg, center_ra_deg, center_dec_deg)

            # appiattiamo anche x e y
            x = np.rad2deg(lon).ravel()
            y = np.rad2deg(lat).ravel()

            data_map, count_map = _do_binning(
                x, y, values,
                bins=[npix_x, npix_y],
                range_=[(x_bins[0], x_bins[-1]), (y_bins[0], y_bins[-1])])

            wcs = self._make_wcs("RA--", "DEC-", center_ra_deg, center_dec_deg, npix_x, npix_y, pixel_size_deg)
            return data_map, count_map, wcs

        x = np.rad2deg(lon).ravel()
        y = -np.rad2deg(lat).ravel()

        x_min = -(npix_x // 2) * pixel_size_deg
        x_max = +(npix_x // 2) * pixel_size_deg
        y_min = -(npix_y // 2) * pixel_size_deg
        y_max = +(npix_y // 2) * pixel_size_deg

        data_map, count_map = _do_binning(
            x, y, values,
            bins=[npix_x, npix_y],
            range_=[(x_min, x_max), (y_min, y_max)])

        wcs = self._make_wcs("AZ--", "EL--", 0, 0, npix_x, npix_y, pixel_size_deg)

        return data_map, count_map, wcs


if __name__ == "__main__":

    base_path = Path(__file__).parents[3]
    config_path = base_path / "configs" / sys.argv[1]

    output_path = base_path / "data" / "output"
    output_path.mkdir(exist_ok=True)

    if not config_path.exists() or config_path.suffix != ".yaml":
        raise ValueError(f"Config file {config_path} does not exist or is not a YAML file")

    config = load_config(config_path)
    scan_ra = Scan.from_dir(config.paths.ra_dir, config.scan)
    # scan_dec = Scan.from_dir(config.paths.ra_dir, config.scan)
    config.calibration.path = next(config.paths.gain_dir.iterdir(), None)
    scan_ra.process(config.calibration, config.filtering)
    # scan_dec.process(config.calibration, config.filtering)

    binner_mm = BinnerMapMaker(scans=[scan_ra], pixel_size=config.map_making.pixel_size, npix=config.map_making.npix)
    data_map_, count_map_, wcs_ = binner_mm.make_map()

    fig, ax = plot_map(
        data_map_,
        count_map_,
        wcs_,
        title="",
        colorbar_label="Phases [rad]",
        savepath=output_path / (config_path.stem + "map.png"))

    plt.show()
