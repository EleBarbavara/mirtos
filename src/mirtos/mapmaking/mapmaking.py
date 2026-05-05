import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d

from mirtos.core.type_defs.scan import Scan
from mirtos.core.projections import conv_radec_to_latlon
from mirtos.core.type_defs.calibration import CalibrationType
from mirtos.core.type_defs.config import MapMakingFrame, load_config
from mirtos.core.type_defs.mapmaking import MapMakingProjection
from mirtos.plotting.map import plot_map, plot_tris_maps
from mirtos.io.fits import to_fits


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
    """Bin samples on a 2D grid and return map, counts, and per-pixel scatter.

    Parameters
    ----------
    x : np.ndarray
        Sample coordinates along the first map axis, expressed in degrees.
    y : np.ndarray
        Sample coordinates along the second map axis, expressed in degrees.
    values : np.ndarray
        One-dimensional array containing the TOD samples to be accumulated in the
        map pixels.
    bins : sequence of int
        Number of bins along the two map axes.
    range_ : sequence of tuple[float, float]
        Lower and upper bounds of the map domain along each axis.

    Returns
    -------
    data_map : np.ndarray
        Mean signal map. Each pixel contains the arithmetic mean of all TOD
        samples that fall into that pixel.
    count_map : np.ndarray
        Hit map. Each pixel contains the number of TOD samples accumulated in
        that pixel.
    std_map : np.ndarray
        Per-pixel standard-deviation map. Each pixel contains the sample
        standard deviation of the TOD values falling into that pixel, i.e. a
        measure of the local scatter of the data contributing to that map bin.
        This is not the uncertainty on the mean map value, but the dispersion of
        the samples inside the pixel.

    Notes
    -----
    The standard deviation is computed independently in each pixel from the
    first two sample moments accumulated during the binning step:

    s_p^2 = \\frac{\\sum x_i^2 - (\\sum x_i)^2 / N_p}{N_p - 1}

    where `N_p` is the number of samples in `p`. The returned
    ``std_map`` is `s_p = \\sqrt{s_p^2}`.

    Pixels with no samples are set to ``NaN`` in ``data_map`` and ``std_map``.
    Pixels with only one sample are set to ``NaN`` in ``std_map`` because the
    sample standard deviation is undefined for :math:`N_p < 2`.
    """
    sum_map, *_ = binned_statistic_2d(
        x, y,
        values=values,
        statistic="sum",
        bins=bins,
        range=range_)

    sumsq_map, *_ = binned_statistic_2d(
        x, y,
        values=values ** 2,
        statistic="sum",
        bins=bins,
        range=range_)

    count_map, *_ = binned_statistic_2d(
        x, y,
        values=values,
        statistic="count",
        bins=bins,
        range=range_)

    data_map = np.full_like(sum_map, np.nan, dtype=float)
    valid = count_map > 0
    data_map[valid] = sum_map[valid] / count_map[valid]

    std_map = np.full_like(sum_map, np.nan, dtype=float)
    valid_std = count_map > 1
    variance = np.zeros_like(sum_map, dtype=float)
    variance[valid_std] = (
        sumsq_map[valid_std] - (sum_map[valid_std] ** 2) / count_map[valid_std]
    ) / (count_map[valid_std] - 1)
    variance[valid_std] = np.maximum(variance[valid_std], 0.0)
    std_map[valid_std] = np.sqrt(variance[valid_std])

    return data_map, count_map, std_map



@dataclass
class MapResult:
    """Container for the maps produced by the map-making step.

    Attributes
    ----------
    data_map : np.ndarray
        Final map obtained by binning the TOD samples and averaging the values
        that fall in each pixel.
    count_map : np.ndarray
        Number of samples accumulated in each map pixel.
    wcs : WCS
        World Coordinate System associated with the returned maps.
    std_map : np.ndarray | None, optional
        Per-pixel standard deviation of the TOD samples contributing to each
        map pixel. It quantifies the intra-pixel scatter of the data and should
        not be interpreted as the error on the mean unless it is further scaled
        by the appropriate factor, e.g. ``1 / sqrt(N)``.
    meta : dict[str, str | float | int]
        Additional metadata to be stored together with the map products.
    """
    data_map: np.ndarray
    count_map: np.ndarray
    wcs: WCS
    std_map: np.ndarray | None = None
    meta: dict[str, str | float | int] = field(default_factory=dict)

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
        """Create a map product from the input scan.

        Returns
        -------
        MapResult
            Object containing the mean signal map (``data_map``), the hit map
            (``count_map``), the associated WCS, and the optional per-pixel
            standard-deviation map (``std_map``).

        Notes
        -----
        The returned ``std_map`` measures the scatter of the TOD samples that
        fall into each pixel after projection and binning. It is therefore a
        local dispersion map, not a map of the uncertainty on the mean signal.
        """
        ...


class BinnerMapMaker(MapMaker):

    def combine_scans(self):
        raise NotImplementedError

    def _make_wcs(self, ctype1: str, ctype2: str, crval1: float, crval2: float,
                  npix_x: int, npix_y: int, pixel_size_deg: float) -> WCS:
        if self.projection == MapMakingProjection.SIN:
            proj = 'SIN'
        elif self.projection == MapMakingProjection.GNOM:
            proj = 'GNOM'
        elif self.projection == MapMakingProjection.EQ:
            proj = 'EQ'
        else:
            ValueError('WCS projection not valid. Enter a valid projection in config file.')
        wcs_dict = {
            "CTYPE1": f"{ctype1}{proj}",
            "CUNIT1": "deg",
            "CDELT1": -pixel_size_deg,
            "CRPIX1": (npix_x + 1) / 2,
            "CRVAL1": crval1,
            "NAXIS1": npix_x,
            "CTYPE2": f"{ctype2}{proj}",
            "CUNIT2": "deg",
            "CDELT2": pixel_size_deg,
            "CRPIX2": (npix_y + 1) / 2,
            "CRVAL2": crval2,
            "NAXIS2": npix_y,
        }
        return WCS(wcs_dict)

    def _make_result(self,
                     data_map: np.ndarray,
                     count_map: np.ndarray,
                     std_map: np.ndarray | None,
                     wcs: WCS) -> MapResult:

        npix_x, npix_y = self.npix
        return MapResult(
            data_map=data_map,
            count_map=count_map,
            std_map=std_map,
            wcs=wcs,
            meta={
                "FRAME": str(self.frame),
                "PROJ": str(self.projection),
                "PIXSIZE": float(self.pixel_size.to_value("arcsec")),
                "NPIXX": npix_x,
                "NPIXY": npix_y,
            },
        )

    def make_map(self):
        npix_x, npix_y = self.npix
        pixel_size_deg = self.pixel_size.to_value('deg')
        center_ra_deg = np.rad2deg(self.ra_center)
        center_dec_deg = np.rad2deg(self.dec_center)

        scan_ = self.scans[0]
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

            x = np.rad2deg(lon).ravel()
            y = np.rad2deg(lat).ravel()

            data_map, count_map, std_map = _do_binning(
                x, y, values,
                bins=[npix_x, npix_y],
                range_=[(x_bins[0], x_bins[-1]), (y_bins[0], y_bins[-1])])

            wcs = self._make_wcs("RA--", "DEC-", center_ra_deg, center_dec_deg, npix_x, npix_y, pixel_size_deg)
            return self._make_result(data_map, count_map, std_map, wcs)

        x = np.rad2deg(lon).ravel()
        y = -np.rad2deg(lat).ravel()

        x_min = -(npix_x // 2) * pixel_size_deg
        x_max = +(npix_x // 2) * pixel_size_deg
        y_min = -(npix_y // 2) * pixel_size_deg
        y_max = +(npix_y // 2) * pixel_size_deg

        data_map, count_map, std_map = _do_binning(
            x, y, values,
            bins=[npix_x, npix_y],
            range_=[(x_min, x_max), (y_min, y_max)])

        wcs = self._make_wcs("AZ--", "EL--", 0, 0, npix_x, npix_y, pixel_size_deg)

        return self._make_result(data_map, count_map, std_map, wcs)


if __name__ == "__main__":

    base_path = Path(__file__).parents[3]
    config_path = base_path / "configs" / sys.argv[1]

    if not config_path.exists() or config_path.suffix != ".yaml":
        raise ValueError(f"Config file `{config_path}` does not exist or is not a YAML file")

    config = load_config(config_path)
    print('Loaded configuration file: ', config_path)

    for scan_path in ["scan_x_dir", "scan_y_dir"]:

        path = getattr(config.paths, scan_path)

        if path is None or not path.exists():
            continue

        fits_files = list(path.rglob("*.fits"))
        if not fits_files:
            continue

        scan = Scan.from_dir(path, config.scan)
        cal_path = next(config.paths.calibration_dir.iterdir(), None) if config.paths.calibration_dir is not None else None
        config.calibration.path = cal_path
        print('Processing TODs.')
        if config.calibration.type == CalibrationType.SKYDIP:
            print('   Calibrating the TODs with skydip.')
        print('   Filtering the TODs.')
        scan.process(config.calibration, config.filtering)

        
        print('Making map.')
        binner_mm = BinnerMapMaker(scans=[scan], pixel_size=config.map_making.pixel_size, npix=config.map_making.npix)
        product = binner_mm.make_map()

        prefix = config_path.stem + " " + scan_path.split('_')[1]
        to_fits(config.paths.output / (prefix + "_map.fits"), product)

        '''
        fig, axes = plot_tris_maps(
            product.data_map,
            product.count_map,
            product.wcs,
            std_map=product.std_map,
            title="",
            colorbar_label="Phases [rad]",
            savepath=config.paths.output / (prefix + "_map.png"),
            dpi=600)
        '''
        
        fig, ax = plot_map(product.data_map, 
                           config, 
                           save_map=False, 
                           wcs=product.wcs
                           )
        
        plt.show()
