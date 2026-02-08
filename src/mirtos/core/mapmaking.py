from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d

from mirtos.core.projections import conv_xy_to_latlon, proj_radec_to_xy
from mirtos.core.types.scan import Scan
from mirtos.core.types.config import MapMakingFrame, load_config


# def make_wcs(ctype1, ctype2, crval1, crval2):
#     wcs_dict = {
#         "CTYPE1": f"{ctype1}{projection}",
#         "CUNIT1": "deg",
#         "CDELT1": -pixel_size_deg,
#         "CRPIX1": npix_x / 2,
#         "CRVAL1": crval1,
#         "NAXIS1": npix_x,
#         "CTYPE2": f"{ctype2}{projection}",
#         "CUNIT2": "deg",
#         "CDELT2": pixel_size_deg,
#         "CRPIX2": npix_y / 2,
#         "CRVAL2": crval2,
#         "NAXIS2": npix_y,
#     }
#     return WCS(wcs_dict)

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

    def make_map(self):

        if len(self.scans) > 1:
            raise ValueError('More than one scan for mapmaking. Run .combine_scans() first')

        npix_x, npix_y = self.npix
        pixel_size_deg = self.pixel_size.to_value('deg')

        center_ra_deg = np.rad2deg(self.ra_center)
        center_dec_deg = np.rad2deg(self.dec_center)

        scan = self.scans[0]

        def make_bins():

            x_bins = np.linspace(
                center_ra_deg - (npix_x // 2) * pixel_size_deg,
                center_ra_deg + (npix_x // 2) * pixel_size_deg,
                npix_x)

            y_bins = np.linspace(
                center_dec_deg - (npix_x // 2) * pixel_size_deg,
                center_dec_deg + (npix_x // 2) * pixel_size_deg,
                npix_y)

            return x_bins, y_bins

        def make_wcs(ctype1, ctype2, crval1, crval2):
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
                "NAXIS2": npix_y}

            return WCS(wcs_dict)

        def do_binning(x, y, bins, range_):
            data_map, *_ = binned_statistic_2d(x, y,
                                               values=scan.tods,
                                               statistic="mean",
                                               bins=bins,
                                               range=range_)

            count_map, *_ = binned_statistic_2d(x, y,
                                                values=scan.tods,
                                                statistic="count",
                                                bins=bins,
                                                range=range_)

            return data_map, count_map

        x_bins, y_bins = make_bins()

        lon, lat = conv_xy_to_latlon(
            *proj_radec_to_xy(scan.ra,
                              scan.dec,
                              center_ra_deg,
                              center_dec_deg,
                              scan.ctx.projection),
            scan.par_angle,
            scan.az,
            scan.el,
            center_ra_deg,
            center_dec_deg,
            scan.ctx.frame)

        if self.frame == MapMakingFrame.RADEC:
            x = np.rad2deg(lat)
            y = np.rad2deg(lon)

            data_map, count_map = do_binning(x, y,
                                             bins=[npix_y, npix_x],
                                             range_=((y_bins[0], y_bins[-1]), (x_bins[0], x_bins[-1])))

            wcs = make_wcs("RA--", "DEC-", center_ra_deg, center_dec_deg)

        elif self.frame == MapMakingFrame.AZEL:

            x = np.rad2deg(lat)
            y = -np.rad2deg(lon)

            print(x.shape, y.shape)

            data_map, count_map = do_binning(x, y,
                                             bins=[npix_x, npix_y],
                                             range_=((x_bins[0] - center_ra_deg, x_bins[-1] - center_ra_deg),
                                                     (y_bins[0] - center_dec_deg, y_bins[-1] - center_dec_deg)))

            wcs = make_wcs("AZ--", "EL--", 0, 0)

        else:
            raise ValueError(f'Frame {self.frame} not supported')

        return data_map, count_map, wcs


if __name__ == "__main__":
    base_path = Path(__file__).parents[3]

    scan_dir = base_path / "data/input/"
    config_path = base_path / "configs/config.yaml"

    config = load_config(config_path)
    scan = Scan.from_dir(scan_dir, config.scan)
    scan.process(config.calibration, config.filtering)

    binner_mm = BinnerMapMaker(scans=[scan], pixel_size=config.map_making.pixel_size, npix=config.map_making.npix)
    data_map_, count_map_, wcs_ = binner_mm.make_map()

    print(data_map_.shape)
