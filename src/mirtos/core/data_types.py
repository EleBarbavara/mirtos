from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from mirtos.core.config_types import Config
import mirtos.io as mirtos_io
from mirtos.mapmaking.mapmaking import rot

import numpy as np
from scipy.constants import c
import yaml

import logging

@dataclass
class Telescope():
    """Generic description of a single-dish telescope."""
    name: str = ''
    diameter: float = 0.
    central_freq: Optional[float] = None
    central_wavelength: Optional[float] = None
    bandwidth: Optional[float] = None
    efficiency: float = 1.0
    fov: Optional[float] = None
    beam: Optional[float] = None
    A_beam: Optional[float] = None
    A_tele: Optional[float] = None
    
    def _end_band(self):
        return self.central_freq + self.bandwidth
    def _beam(self): 
        #beam    -> beam gaussiano : beam/(2*np.log(2))
        return (self.central_freq/self.diameter)
    def _A_beam(self):
        return np.pi*self.beam**2/(4*np.log(2))
    def _A_tele(self):
        #unit: m2
        return np.pi*((self.diameter/2)**2)

    def initialize(self, config: Config) -> None:
        with open(config.paths.instrumentation, "r") as f:
            cfg_instr = yaml.safe_load(f)
        
        if config.telescope in cfg_instr:
            cfg_instr = cfg_instr[config.telescope]
            
            self.name = config.telescope
            if 'note' in cfg_instr:
                raise ValueError('Processing and mapmaker for '+self.name+' are not currently possible.')
            elif cfg_instr['diameter']>1:
                self.diameter = float(cfg_instr['diameter'])
                self.bandwidth = float(cfg_instr['bandwidth'])
                self.central_wavelength = float(cfg_instr['central_wavelength'])
                self.central_freq = float(cfg_instr['central_freq'])
                self.efficiency = float(cfg_instr['efficiency'])
                self.fov = float(cfg_instr['fov'])
                self.end_band = self._end_band()
                self.beam = self._beam()
                self.A_beam = self._A_beam()
                self.A_tele= self._A_tele()
            else:
                self.diameter = float(cfg_instr['diameter'])
                self.bandwidth = float(cfg_instr['bandwidth'])
                self.central_wavelength = float(cfg_instr['central_wavelength'])
                self.central_freq = float(cfg_instr['central_freq'])
                self.efficiency = float(cfg_instr['efficiency'])
                self.end_band = self._end_band()
                self.beam = float(cfg_instr['beam'])
                self.A_beam = float(cfg_instr['A_beam'])
                self.A_tele= float(cfg_instr['A_tele'])
                self.fov = float(cfg_instr['fov'])
  
    
@dataclass
class Subscan():
    telescope: str = ''
    dati_beammap : pd.core.frame.DataFrame = field(default_factory=dict)
    
    def load_telescope(self, cfg: Config):
        '''
        mode = radiotelescope with which the observation was carried out.
        
        Available mode:
        - 'Lab'
        - 'SRT'
        - 'GBT'
        '''
        system = cfg.system
        
        self.telescope = Telescope()
        self.telescope.initialize(config= cfg)
        
        logging.info("-------------------------------------")
        logging.info("         Telescope: "+self.telescope.name+"  ")
        logging.info("-------------------------------------")
        logging.info("Diameter = "+str(self.telescope.diameter)+" m")
        logging.info("Band = "+str(self.telescope.central_freq/1e9)+"-"+str(self.telescope.end_band/1e9)+" GHz")
        logging.info("Efficiency = "+str(self.telescope.efficiency)+" ")
        logging.info("Gaussian beam = ????????????? ")
        logging.info("Beam area = ???????????? ")
        logging.info("-------------------------------------")
    
    def load_pixel_offsets(self, cfg : Config):

        if cfg.paths.offset_det!=False:
            dati = np.genfromtxt(cfg.paths.offset_det, comments='#')
            self.dati_beammap = pd.DataFrame(dati, columns=['id', 'lon-offset', 'lat-offset', 'Tcal pol1', 'Tcal pol2', 'flag'])#, 'peak']) #flag: 0=>ok, 1=>bad
            #IMPORTO GLI OFFSET DEI FEED

        
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
        logging.debug("Noffsets = "+str(np.shape(offset_x))+"\n"+"Nfeeds="+str(num_feed)+"\n"+"Nexcl="+str(excl_feed))
        
        return offset_x, offset_y, tod, num_feed, excl_feed
        
        
    def extract_data(self, filename, config):
        '''
        This function extract the metadata from the fits file, normalize all the tods for the optical responsivity
        and create the ra, dec and tod arrays of the observation.
        '''
        
        if self.mode == 'SRT':
            
            hdul = mirtos_io.fits.read_discos_fits(self, filename, config)
            
            if config.paths.offset_det != False:
                self.xOffset, self.yOffset, self.tod_raw, self.num_feed, self.excl_feed = self.exclude_channels(self.tod_raw, self.num_feed, self.dati_beammap)
                
            else:

                self.xOffset = np.deg2rad(hdul['FEED TABLE'].data['xOffset'])
                self.yOffset = np.deg2rad(hdul['FEED TABLE'].data['yOffset'])
            
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