import numpy as np
from astropy.io import fits
import pandas as pd
#from astropy.table import Table
#from astropy.stats import sigma_clip
#from scipy.optimize import curve_fit
#from scipy.signal import medfilt
#from scipy.signal import detrend
#from astropy.coordinates import EarthLocation, Angle
#import datetime
#from astropy.time import Time
#from astropy import units as u
#from astropy.convolution import convolve, Gaussian2DKernel
#from lmfit import Model, Parameters
#from astropy.visualization import simple_norm
import lib
from telescope_class import Telescope


class Subscan():
    
    def __init__(self, cfg, mode='SRT'):
        '''
        mode = radiotelescope with which the observation was carried out.
        
        Available mode:
        - 'SRT'
        - 'GBT'
        '''
        self.mode = mode
        self.all_range_timestep = []
        self.num_feed_all = []
        
        if cfg['paths']['offset_pixel']!=False:
            dati = np.genfromtxt(cfg['paths']['offset_pixel'], comments='#')
            self.dati_beammap = pd.DataFrame(dati, columns=['id', 'lon-offset', 'lat-offset', 'Tcal pol1', 'Tcal pol2', 'flag'])#, 'peak']) #flag: 0=>ok, 1=>bad
            #IMPORTO GLI OFFSET DEI FEED
        
    def init_tele(self):    
        if self.mode == 'SRT':
            self.tele = Telescope(self.mode)
            
            print("-------------------------------------")
            print("         Telescope: "+self.mode+"  ")
            print("-------------------------------------")
            print("Diameter = "+str(self.tele.D_SRT)+" m")
            print("Band = "+str(self.tele.nu/1e9)+"-"+str(self.tele.fine_band/1e9)+" GHz")
            print("Efficiency = "+str(self.tele.eff)+" ")
            print("Gaussian beam = ????????????? ")
            print("Beam area = ???????????? ")
            print("-------------------------------------")
        
        elif self.mode=='GBT':
            print('Observations with GBT are not currently possible.')
        
        else:
            raise ValueError('Select radiotelescope.')
        
        
        
    def extract_data(self, filename, config):
        '''
        This function extract the metadata from the fits file, normalize all the timestream for the optical responsivity
        and create the ra, dec and tod arrays of the observation.
        '''
        
        if self.mode == 'SRT':
            
            hdul = fits.open(filename) 
            
            #self.srp_tz = np.mean(hdul['SERVO TABLE'].metadata['SRP_TZ']) #mm
            #self.srp_ty = np.mean(hdul['SERVO TABLE'].metadata['SRP_TY']) #mm
            #self.srp_tx = np.mean(hdul['SERVO TABLE'].metadata['SRP_TX']) #mm
            ft = hdul['DATA TABLE'].data['flag_track']
            
            if config['flag_track'] == True:
                flag_track = ft.astype(bool)
            else:
                flag_track = np.ones(len(ft)).astype(bool)
            
            ft = hdul['DATA TABLE'].data['flag_track']
            flag_track = ft > 0.5
            
            try: 
                angle_offset = 0 #np.pi/2
                par_angle_nan = hdul['DATA TABLE'].data['par_angle'][flag_track]
                self.par_angle = par_angle_nan[np.logical_not(np.isnan(par_angle_nan))] + angle_offset
                ra_scan_nan = hdul['DATA TABLE'].data['raj2000'][flag_track] 
                self.ra_scan = ra_scan_nan[np.logical_not(np.isnan(ra_scan_nan))]
                dec_scan_nan = hdul['DATA TABLE'].data['decj2000'][flag_track] 
                self.dec_scan = dec_scan_nan[np.logical_not(np.isnan(dec_scan_nan))]
                az_scan_nan = hdul['DATA TABLE'].data['az'][flag_track] 
                self.az_scan = az_scan_nan[np.logical_not(np.isnan(az_scan_nan))]
                el_scan_nan = hdul['DATA TABLE'].data['el'][flag_track] 
                self.el_scan = el_scan_nan[np.logical_not(np.isnan(el_scan_nan))]
                time_all = hdul["DATA TABLE"].data["time"][flag_track] 
                self.time = time_all[np.logical_not(np.isnan(dec_scan_nan))]
                self.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
                self.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
                '''
                angle_offset = 0 #np.pi/2
                self.par_angle = hdul['DATA TABLE'].metadata['par_angle'][flag_track] + angle_offset
                self.ra_scan = hdul['DATA TABLE'].metadata['raj2000'][flag_track] 
                self.dec_scan = hdul['DATA TABLE'].metadata['decj2000'][flag_track]
                self.time = hdul["DATA TABLE"].metadata["time"][flag_track]
                self.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
                self.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
                '''
            except:
                angle_offset = 0#-np.pi/2
                par_angle_nan = hdul['DATA TABLE INTERP'].data['par_angle_interpolated'][flag_track]
                self.par_angle = par_angle_nan[np.logical_not(np.isnan(par_angle_nan))] + angle_offset
                ra_scan_nan = hdul['DATA TABLE INTERP'].data['raj2000_interpolated'][flag_track] 
                self.ra_scan = ra_scan_nan[np.logical_not(np.isnan(ra_scan_nan))]
                dec_scan_nan = hdul['DATA TABLE INTERP'].data['decj2000_interpolated'][flag_track] 
                self.dec_scan = dec_scan_nan[np.logical_not(np.isnan(dec_scan_nan))]
                self.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
                self.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
            
            self.range_timestep = range(0, len(self.par_angle), 1) #(1222, 147704) daisy #(1209, 9647) 30s #where the fits file is not 'NULL' or not zero
            self.all_range_timestep.append(self.range_timestep)
            self.num_timestep = int(len(self.range_timestep))
            self.sample_freq = 244.140625 #256e6/2**20 
            self.tstep = 1/self.sample_freq
            self.nyqfreq = self.sample_freq/2
            
            self.num_feed = len(hdul["PH TABLE"].data[0])
            self.num_feed_all.append(self.num_feed)
            
            hdr = ["chp_"+str(i).zfill(3) for i in range(self.num_feed)]
            
            self.timestream_raw = []
            #self.timestream_raw = hdul['PH TABLE'].metadata[hdr[0]][flag_track]
            for i in range(0, self.num_feed):
                #try:
                #self.timestream_raw.append(hdul['PH TABLE'].metadata[hdr[i]][flag_track])
                ts_nan = hdul['PH TABLE'].data[hdr[i]][flag_track]
                self.timestream_raw.append(ts_nan[np.logical_not(np.isnan(ra_scan_nan))])
                #except:
                #    self.timestream_raw.append([np.nan]*self.num_timestep)
            
            if config['paths']['offset_pixel']!=False:
                self.offset_x_single, self.offset_y_single, self.timestream_raw, self.num_feed, self.excl_feed = lib.exclude_pixel(self.timestream_raw, self.num_feed, self.dati_beammap)
                
                if config['binner']['frame']=='RADEC':
                    self.offset_x = []
                    self.offset_y = []
                    
                    for i in range(self.num_feed):
                                    xoff_rot, yoff_rot = lib.rot([self.offset_x_single[i]] * len(self.par_angle), [self.offset_y_single[i]] * len(self.par_angle), self.par_angle)
                                    self.offset_x.append(xoff_rot)
                                    self.offset_y.append(yoff_rot)
                    
                else:
                    self.offset_x = self.offset_x_single
                    self.offset_y = self.offset_y_single
            else:
                self.offset_x = np.deg2rad(hdul['FEED TABLE'].data['xOffset '])
                self.offset_y = np.deg2rad(hdul['FEED TABLE'].data['yOffset '])
            
            hdul.close()



#GRANDEZZE della CLASSE Observation: (> che si possono chiamare nel codice con 'obs.grandezza')
#file_resp > [str] > file pickle delle responsivita' dei canali dato in entrata alla classe Observation 
#opt_resp > [pandas.core.frame.DataFrame] > dataframe che contiene (channel, Opt_resp, error)
#excl_feed > [numpy.ndarray] > array che contiene i feed esclusi che non hanno responsivita' (contiene il numero identificativo del feed)
#obj > [str] > file fits dell'osservazione dato in entrata alla classe Observation
#name_target > [str] > nome del target dell'osservazione dato in entrata alla classe Observation
#mode > [str] > radiotelescopio usato per l'osservazione
#timestream_raw > [list] > list dei timestream rozzi salvati dal file fits dell'osservazione
#tsdt > [numpy.ndarray] > array dei timestram detrenzizzati 
#ts > [numpy.ndarray] > array dei timestream finali (detrendizzati e rinormalizatti con le responsivita') -> sono uguali a timestream_raw ma sono amplificati rispetto alla responsivita'
#pixel_mask > [list] > lista delle maschere di ogni canale create con il sigma clip sulla sorgente (mascherati i punti che stanno dentro 4 sigma dalla media del segnale+la sorgente)
#num_feed > [int] > numero di canali
#num_timestep > [int] > numero di timestep del timestream
#range_timestep > [range] > range dei timestep
#sample_freq > [float] > sample frequency
#nyqfreq > [float] > Nyquist frequency
#feed_x > [numpy.ndarray] >  coordinata x della posizione dei feed rispetto al feed centrale
#feed_y > [numpy.ndarray] > coordinata y della posizione dei feed rispetto al feed centrale
#par_angle > [numpy.ndarray] > angolo parallattico durante la scansione
#ra_scan > [numpy.ndarray] > tutte le RA durante la scansione (RA del feed centrale)
#dec_scan > [numpy.ndarray] > tutte le Dec durante la scansione (RA del feed centrale)
#ra > [numpy.ndarray] > RA di tutti i feed durante la scansione
#dec > [numpy.ndarray] > Dec di tutti i feed durante la scansione
#totx > [numpy.ndarray] > array delle RA srotolate per poterle plottare
#toty > [numpy.ndarray] > array delle Dec srotolate per poterle plottare
#s > [numpy.ndarray] > array dei timestream srotolati per poterli plottare

