from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
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
    x_bins = np.linspace(x_min, x_max, npix_x)
    y_bins = np.linspace(y_min, y_max, npix_y)
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
            "CRPIX1": npix_x / 2,
            "CRVAL1": crval1,
            "NAXIS1": npix_x,
            "CTYPE2": f"{ctype2}{self.projection}",
            "CUNIT2": "deg",
            "CDELT2": pixel_size_deg,
            "CRPIX2": npix_y / 2,
            "CRVAL2": crval2,
            "NAXIS2": npix_y,
        }
        return WCS(wcs_dict)

    def _compute_lon_lat(self, scan_: Scan, scan_mask: np.ndarray,
                         center_ra_rad: float, center_dec_rad: float):
        return conv_radec_to_latlon(
            scan_.ra[scan_mask],
            scan_.dec[scan_mask],
            center_ra_rad,
            center_dec_rad,
            self.projection,
            scan_.par_angle[scan_mask],
            scan_.ctx.beammap.beam_map['lon_offset'].to_numpy(),
            scan_.ctx.beammap.beam_map['lat_offset'].to_numpy(),
            self.frame)

    def make_map(self):
        npix_x, npix_y = self.npix
        pixel_size_deg = self.pixel_size.to_value('deg')
        center_ra_deg = np.rad2deg(self.ra_center)
        center_dec_deg = np.rad2deg(self.dec_center)

        scan_ = self.scans[0]
        scan_mask = scan_.mask
        values = scan_.calibrated_tods[:, scan_mask].ravel()

        lon, lat = self._compute_lon_lat(scan_, scan_mask, self.ra_center, self.dec_center)

        if self.frame == MapMakingFrame.RADEC:
            x_bins, y_bins = _make_bins(npix_x, npix_y, pixel_size_deg, center_ra_deg, center_dec_deg)

            x = np.rad2deg(lat).ravel()
            y = np.rad2deg(lon).ravel()

            data_map, count_map = _do_binning(
                x, y, values,
                bins=[npix_y, npix_x],
                range_=((y_bins[0], y_bins[-1]), (x_bins[0], x_bins[-1])))

            wcs = self._make_wcs("RA--", "DEC-", center_ra_deg, center_dec_deg, npix_x, npix_y, pixel_size_deg)
            return data_map, count_map, wcs

        x = np.rad2deg(lat).ravel()
        y = -np.rad2deg(lon).ravel()

        x_min = -(npix_y // 2) * pixel_size_deg
        x_max = +(npix_y // 2) * pixel_size_deg
        y_min = -(npix_x // 2) * pixel_size_deg
        y_max = +(npix_x // 2) * pixel_size_deg

        data_map, count_map = _do_binning(
            x, y, values,
            bins=[npix_y, npix_x],
            range_=[(x_min, x_max), (y_min, y_max)])

        wcs = self._make_wcs("AZ--", "EL--", 0, 0, npix_x, npix_y, pixel_size_deg)

        return data_map, count_map, wcs


if __name__ == "__main__":
    base_path = Path(__file__).parents[3]

    #config_path = base_path / "configs" / "a1995_conf.yaml"
    config_path = base_path / "configs" / "cygA_conf.yaml"
    config = load_config(config_path)
    scan_ra = Scan.from_dir(config.paths.ra_dir, config.scan)
    # scan_dec = Scan.from_dir(config.paths.ra_dir, config.scan)
    config.calibration.path = next(config.paths.gain_dir.iterdir())
    scan_ra.process(config.calibration, config.filtering)
    # scan_dec.process(config.calibration, config.filtering)

    binner_mm = BinnerMapMaker(scans=[scan_ra], pixel_size=config.map_making.pixel_size, npix=config.map_making.npix)
    data_map_, count_map_, wcs_ = binner_mm.make_map()

    mask = count_map_ > 0
    vmin, vmax = np.nanpercentile(data_map_[mask], [2, 98])
    ys, xs = np.where(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    fig, ax = plot_map(data_map_, wcs_, None, vmin=vmin, vmax=vmax)
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    ax.coords.grid(False)
    cbar = fig.colorbar(ax.images[0], ax=ax, shrink=0.6, pad=0.03)

    cbar.set_label("Phases [rad]")
    fig.savefig("test_" + ("ra.png" if "scan_ra" in locals() else "dec.png"), dpi=1000)
    fig.show()
