import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.wcs import WCS
from src.mirtos import lib


class Binner():
        def __init__(self, bin_mode, projection='SIN'):
                '''
                bin_mode = binning map along which coordinates?
                
                Available frame:
                - 'RADEC'
                - 'AZEL'
                '''
                self.mode = bin_mode
                self.projection = projection
                self.wcs = 0
        
        def proj_radec_to_xy(self, ra,dec,ra0,dec0, projection):

                if projection=='SIN':
                        lam = ra
                        phi = dec
                        #phi1 = self.scan_center_ra
                        lam0 = ra0

                        x = (lam-lam0)*np.cos(phi) + lam0
                        y = phi
                        
                        return x, y
                
                if projection=='GNOM':
                         #https://mathworld.wolfram.com/GnomonicProjection.html
                        phi = dec
                        lam = ra
                        phi1 = dec0
                        lam0 = ra0

                        c = np.sin(phi1)*np.sin(phi) + np.cos(phi1)*np.cos(phi)*np.cos(lam-lam0)
                        
                        self.x = (np.cos(phi) * np.sin(lam-lam0))/c
                        self.y = (np.cos(phi1)*np.sin(phi) - np.sin(phi1)*np.cos(phi)*np.cos(lam-lam0))/c
                                        
                        return x, y
        
                else:
                        raise ValueError(projection, ': this projection not available.')
        
        def conv_xy_to_latlon(self, x, y, par_angle, num_feed, offset_x, offset_y,  center_ra, center_dec):
                self.lon = []
                self.lat = []
                
                if self.mode == 'RADEC':
                        '''  
                        xoff_rot = []
                        yoff_rot = []
                        for i in range(num_feed):
                                x_rot, y_rot = lib.rot([offset_x[i]]*len(par_angle), [offset_y[i]]*len(par_angle), par_angle)
                                xoff_rot.append(x_rot)
                                yoff_rot.append(y_rot)
                        print(np.shape(xoff_rot)) 
                        '''     
                                
                        for i in range(num_feed):        
                                self.lat.append(y + offset_y[i])
                                self.lon.append(x - offset_x[i])#/np.cos(y + offset_y[i]))
                         
                
                elif self.mode == 'AZEL':
                        
                        x_rot, y_rot = lib.rot(x - center_ra, y - center_dec, par_angle)
                        
                        for i in range(num_feed):
                                try:
                                        self.lat.append(y_rot + offset_y[i])
                                        self.lon.append(x_rot - offset_x[i]/np.cos(y_rot + offset_y[i]))
                                except:
                                        self.lat.append(y_rot)
                                        self.lon.append(x_rot/np.cos(y_rot))

                elif self.mode == 'EQ':
                        
                        #x_rot, y_rot = lib.rot(x-center_ra, y-center_dec, par_angle)
                        x_shift = x-center_ra
                        y_shift = y-center_dec
                        
                        for i in range(num_feed):
                                self.lat.append(y_shift + offset_y[i])
                                self.lon.append(x_shift - offset_x[i]/np.cos(y_shift + offset_y[i]))
                
                else:      
                        raise ValueError(self.mode + '-> this set of coordinates is not available.')
                
                
                
        def bin_map(self, lat, lon, tods, center_ra, center_dec, npix_x=128, npix_y=128, pixel_size_deg=4/3600, projection="SIN"):
                
                if self.mode == 'RADEC':
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
                        
                
                elif self.mode == 'AZEL':
                                        
                        ### XY must be projected!

                        pixel_size = pixel_size_deg #converting all to radians
                        center_ra_deg = np.rad2deg(center_ra)
                        center_dec_deg = np.rad2deg(center_dec)

                        self.x_bins = np.linspace(center_ra_deg - npix_x//2 * pixel_size, center_ra_deg+npix_x//2 * pixel_size, npix_x)
                        self.y_bins = np.linspace(center_dec_deg - npix_x//2 * pixel_size, center_dec_deg+npix_x//2 * pixel_size, npix_y)

                        map_width = self.x_bins[-1] - self.x_bins[0]
                        map_height = self.y_bins[-1] - self.y_bins[0]

                        #print("NAXIS1=",self.x_bins.shape)
                        #print("NAXIS2=",self.y_bins.shape)
                        #print("width=",map_width)
                        #print("height=", map_height)

                        
                        
                        data_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                                -np.rad2deg(lon),
                                                                                values = tods,
                                                                                statistic="mean",
                                                                                bins= [npix_x, npix_y],
                                                                                range = ((self.x_bins[0]-center_ra_deg, self.x_bins[-1]-center_ra_deg), 
                                                                                         (self.y_bins[0]-center_dec_deg, self.y_bins[-1]-center_dec_deg)))

                        count_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                                -np.rad2deg(lon),
                                                                                values = tods,
                                                                                statistic="count",
                                                                                bins= [npix_x, npix_y],
                                                                                range = ((self.x_bins[0]-center_ra_deg, self.x_bins[-1]-center_ra_deg),
                                                                                         (self.y_bins[0]-center_dec_deg, self.y_bins[-1]-center_dec_deg)))


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
                
                elif self.mode == 'EQ':
                                        
                        ### XY must be projected!

                        pixel_size = pixel_size_deg #converting all to radians
                        center_ra_deg = np.rad2deg(center_ra)
                        center_dec_deg = np.rad2deg(center_dec)

                        self.x_bins = np.linspace(center_ra_deg - npix_x//2 * pixel_size, center_ra_deg+npix_x//2 * pixel_size, npix_x)
                        self.y_bins = np.linspace(center_dec_deg - npix_x//2 * pixel_size, center_dec_deg+npix_x//2 * pixel_size, npix_y)

                        map_width = self.x_bins[-1] - self.x_bins[0]
                        map_height = self.y_bins[-1] - self.y_bins[0]

                        #print("NAXIS1=",self.x_bins.shape)
                        #print("NAXIS2=",self.y_bins.shape)
                        #print("width=",map_width)
                        #print("height=", map_height)

                        
                        
                        data_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                                -np.rad2deg(lon),
                                                                                values = tods,
                                                                                statistic="mean",
                                                                                bins= [npix_x, npix_y],
                                                                                range = ((self.x_bins[0]-center_ra_deg, self.x_bins[-1]-center_ra_deg), 
                                                                                         (self.y_bins[0]-center_dec_deg, self.y_bins[-1]-center_dec_deg)))

                        count_map, x_edge, y_edge, binnumber = binned_statistic_2d(np.rad2deg(lat), 
                                                                                -np.rad2deg(lon),
                                                                                values = tods,
                                                                                statistic="count",
                                                                                bins= [npix_x, npix_y],
                                                                                range = ((self.x_bins[0]-center_ra_deg, self.x_bins[-1]-center_ra_deg),
                                                                                         (self.y_bins[0]-center_dec_deg, self.y_bins[-1]-center_dec_deg)))


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
                
                        

