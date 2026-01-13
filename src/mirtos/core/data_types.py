from dataclasses import dataclass
from typing import Optional
import pandas as pd
import src.mirtos.io as mirtos_io
from src.mirtos.mapmaking.mapmaking import rot

import numpy as np
from scipy.constants import c


@dataclass
class Telescope:
    """Generic description of a single-dish telescope."""
    name: str
    diameter_m: float
    central_freq_hz: Optional[float] = None
    central_wavelength_m: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    efficiency: float = 1.0
    fov_arcmin: Optional[float] = None
    beam: float
    A_beam: float
    A_tele: float
    
    def __post_init__(self) -> None:
        if self.name == 'SRT':
            self.diameter_m = 60 #m
            self.bandwidth_hz = 30e9 #Hz
            self.central_wavelength_m = 0.0033 #m (central wavelenght)
            self.central_freq_hz = 90e9 #GHz (central frequency)
            self.end_band = self.central_freq_hz+self.bandwidth_hz
            self.efficiency = 0.3
            self.beam = (self.central_freq_hz/self.diameter_m) #beam    -> beam gaussiano = beam/(2*np.log(2))
            self.A_beam = np.pi*self.beam**2/(4*np.log(2))
            self.A_tele= np.pi*((self.diameter_m/2)**2) #m2
            self.fov_arcmin = 4 #arcmin
            
        elif self.name == 'LAB':
            self.diameter_m = 0 #m
            self.bandwidth_hz = 30e9 #Hz
            self.central_wavelength_m = 0.0033 #m (central wavelenght)
            self.central_freq_hz = 90e9 #GHz (central frequency)
            self.end_band = self.central_freq_hz+self.bandwidth_hz
            self.efficiency = 0.3
            self.beam = 0
            self.A_beam = 0
            self.A_tele= 0
            self.fov_arcmin = 4 #arcmin
        
        elif self.name=='GBT':
            raise ValueError('Observation with this telescope are not currently possible.')
            
    
@dataclass
class Subscan():
    telescope: str = ''
    dati_beammap : pd.core.frame.DataFrame = {}
    
    def initialize(self, cfg: dict, telescope: str = 'SRT'):
        '''
        mode = radiotelescope with which the observation was carried out.
        
        Available mode:
        - 'Lab'
        - 'SRT'
        - 'GBT'
        '''
        self.telescope = telescope
        
        if cfg['paths']['offset_det']!=False:
            dati = np.genfromtxt(cfg['paths']['offset_det'], comments='#')
            self.dati_beammap = pd.DataFrame(dati, columns=['id', 'lon-offset', 'lat-offset', 'Tcal pol1', 'Tcal pol2', 'flag'])#, 'peak']) #flag: 0=>ok, 1=>bad
            #IMPORTO GLI OFFSET DEI FEED
        
    def init_telescope(self):    
        if self.telescope == 'SRT':
            self.tele = Telescope(self.telescope)
            
            print("-------------------------------------")
            print("         Telescope: "+self.telescope+"  ")
            print("-------------------------------------")
            print("Diameter = "+str(self.tele.diameter_m)+" m")
            print("Band = "+str(self.tele.central_freq_hz/1e9)+"-"+str(self.tele.end_band/1e9)+" GHz")
            print("Efficiency = "+str(self.tele.efficiency)+" ")
            print("Gaussian beam = ????????????? ")
            print("Beam area = ???????????? ")
            print("-------------------------------------")
        
        elif self.telescope=='GBT':
            print('Observations with GBT are not currently possible.')
        
        else:
            raise ValueError('Select radiotelescope.')
        
    def exclude_channels(tod, num_feed, dati_beammap):
        excl_feed = np.empty(0)
        for i in range(len(dati_beammap['flag'])):
            if dati_beammap['flag'][i]==2: #if dati_beammap['flag'][i]==1:
                excl_feed = np.append(excl_feed, int(i))
        excl_feed = excl_feed.astype(int)

        #controllo che nel file degli offset sono inseriti tutti i pixel 
        if len(dati_beammap['flag'])<num_feed:
            add_excluded = np.arange(len(dati_beammap['flag']), num_feed)
            excl_feed = np.concatenate((excl_feed, add_excluded))
        excl_feed = np.sort(excl_feed)[::-1] #li sorto al contratio perche quando vado a eliminare i ts, se parto dal primo si modificano i numeri del canale e non elimino più quelli che vorrei
        
        flag = dati_beammap['flag']<3 #<1
        offset_x = list(np.deg2rad(dati_beammap['lon-offset'][flag]))
        offset_y = list(np.deg2rad(dati_beammap['lat-offset'][flag]))
        #offset_x = list(np.deg2rad(dati_beammap['az off'][flag]))
        #offset_y = list(np.deg2rad(dati_beammap['el off'][flag]))
        
        #Dropping the KIDs with no responsivity
        tod = list(tod)
        count = 0
        
        for i in excl_feed:
            if  len(tod)-1<i:
                #print('No channel ', i)
                count +=1
            else:
                tod.pop(i) # = np.delete(tod, i, axis=0)
            #pixel_mask.pop(i) # = np.delete(pixel_mask, i, axis=0)
        num_feed -= (len(excl_feed)-count)
        print(np.shape(offset_x), num_feed, excl_feed)
        
        return offset_x, offset_y, tod, num_feed, excl_feed
        
        
    def extract_data(self, filename, config):
        '''
        This function extract the metadata from the fits file, normalize all the timestream for the optical responsivity
        and create the ra, dec and tod arrays of the observation.
        '''
        
        if self.mode == 'SRT':
            
            hdul = mirtos_io.fits.read_discos_fits(self, filename, config)
            
            if config['paths']['offset_det']!=False:
                self.offset_x_single, self.offset_y_single, self.timestream_raw, self.num_feed, self.excl_feed = self.exclude_channels(self.timestream_raw, self.num_feed, self.dati_beammap)
                
                if config['binner']['frame']=='RADEC':
                    self.offset_x = []
                    self.offset_y = []
                    
                    for i in range(self.num_feed):
                                    xoff_rot, yoff_rot = rot([self.offset_x_single[i]] * len(self.par_angle), [self.offset_y_single[i]] * len(self.par_angle), self.par_angle)
                                    self.offset_x.append(xoff_rot)
                                    self.offset_y.append(yoff_rot)
                    
                else:
                    self.offset_x = self.offset_x_single
                    self.offset_y = self.offset_y_single
            else:
                self.offset_x = np.deg2rad(hdul['FEED TABLE'].data['xOffset '])
                self.offset_y = np.deg2rad(hdul['FEED TABLE'].data['yOffset '])
            
            hdul.close()


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