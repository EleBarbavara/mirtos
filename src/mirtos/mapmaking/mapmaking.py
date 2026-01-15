import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
from astropy.wcs import WCS

#from mirtos.core.data_types import Subscan
#from mirtos.binner import Binner
from mirtos.cleaner_class import Cleaner




def bin_map(frame, lat, lon, tods, center_ra, center_dec, npix_x=128, npix_y=128, pixel_size_deg=4/3600, projection="SIN"):

        if frame == 'RADEC':
                ### XY must be projected!

                pixel_size = pixel_size_deg #converting all to radians
                center_ra_deg = np.rad2deg(center_ra)
                center_dec_deg = np.rad2deg(center_dec)

                x_bins = np.linspace(center_ra_deg - npix_x//2 * pixel_size, center_ra_deg+npix_x//2 * pixel_size, npix_x)
                y_bins = np.linspace(center_dec_deg - npix_x//2 * pixel_size, center_dec_deg+npix_x//2 * pixel_size, npix_y)
                
                map_width = x_bins[-1] - x_bins[0]
                map_height = y_bins[-1] - y_bins[0]
                
                #print("NAXIS1=",self.x_bins.shape)
                #print("NAXIS2=",self.y_bins.shape)
                #print("width=",map_width)
                #print("height=", map_height)

                data_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat),
                                                                        np.rad2deg(lon), 
                                                                        values = tods,
                                                                        statistic="mean",
                                                                        bins= [npix_y, npix_x],
                                                                        range = ((y_bins[0], y_bins[-1]),
                                                                                (x_bins[0], x_bins[-1])))
                count_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                        np.rad2deg(lon),
                                                                        values = tods,
                                                                        statistic="count",
                                                                        bins= [npix_x, npix_y])#,
                                                                        #range = ((self.x_bins[0], self.x_bins[-1]),
                                                                        #         (self.y_bins[0], self.y_bins[-1])))

                ### building the WCS
                wcs_dict = {
                "CTYPE1": "RA--"+projection,
                "CUNIT1": "deg",
                "CDELT1": -pixel_size_deg,
                "CRPIX1": npix_x/2,
                "CRVAL1": center_ra_deg,
                "NAXIS1": npix_x,
                "CTYPE2": "DEC-"+projection,
                "CUNIT2": "deg",
                "CDELT2": pixel_size_deg,
                "CRPIX2": npix_y/2,
                "CRVAL2": center_dec_deg,
                "NAXIS2": npix_y,
                }

                wcs = WCS(wcs_dict)
                

        elif frame == 'AZEL':
                                
                ### XY must be projected!

                pixel_size = pixel_size_deg #converting all to radians
                center_ra_deg = np.rad2deg(center_ra)
                center_dec_deg = np.rad2deg(center_dec)

                x_bins = np.linspace(center_ra_deg - npix_x//2 * pixel_size, center_ra_deg+npix_x//2 * pixel_size, npix_x)
                y_bins = np.linspace(center_dec_deg - npix_x//2 * pixel_size, center_dec_deg+npix_x//2 * pixel_size, npix_y)

                map_width = x_bins[-1] - x_bins[0]
                map_height = y_bins[-1] - y_bins[0]

                #print("NAXIS1=",self.x_bins.shape)
                #print("NAXIS2=",self.y_bins.shape)
                #print("width=",map_width)
                #print("height=", map_height)

                
                
                data_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                        -np.rad2deg(lon),
                                                                        values = tods,
                                                                        statistic="mean",
                                                                        bins= [npix_x, npix_y],
                                                                        range = ((x_bins[0]-center_ra_deg, x_bins[-1]-center_ra_deg), 
                                                                                        (y_bins[0]-center_dec_deg, y_bins[-1]-center_dec_deg)))

                count_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                        -np.rad2deg(lon),
                                                                        values = tods,
                                                                        statistic="count",
                                                                        bins= [npix_x, npix_y],
                                                                        range = ((x_bins[0]-center_ra_deg, x_bins[-1]-center_ra_deg),
                                                                                        (y_bins[0]-center_dec_deg, y_bins[-1]-center_dec_deg)))


                ### building the WCS
                wcs_dict = {
                "CTYPE1": "AZ--"+projection,
                "CUNIT1": "deg",
                "CDELT1": -pixel_size_deg,
                "CRPIX1": npix_x/2,
                "CRVAL1": 0,
                "NAXIS1": npix_x,
                "CTYPE2": "EL--"+projection,
                "CUNIT2": "deg",
                "CDELT2": pixel_size_deg,
                "CRPIX2": npix_y/2,
                "CRVAL2": 0,
                "NAXIS2": npix_y,
                }

                wcs = WCS(wcs_dict)


        return data_map, count_map, wcs

                

