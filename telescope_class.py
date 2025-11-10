import numpy as np
from scipy.constants import c, k

# c = 3e8 #m/s
# k = k_B = 1.38e-23 #J/K


class Telescope():
    def __init__(self, tele=""):
        self.telescope = tele
        
        if self.telescope == 'SRT':
            self.D_SRT = 60 #m
            self.band = 30e9 #Hz
            self.wl = 0.0033 #m (central wavelenght)
            self.nu = 90e9 #GHz (central frequency)
            self.eff = 0.3
            self.beam = (self.wl/self.D_SRT) #beam    -> beam gaussiano = beam/(2*np.log(2))
            self.A_beam = np.pi*self.beam**2/(4*np.log(2))
            self.A_tele= np.pi*((self.D_SRT/2)**2) #m2
            self.FOV = 4 #arcmin
            self.fine_band = self.nu+self.band
        
        elif self.telescope=='GBT':
            raise ValueError('Observation with this telescope are not currently possible.')



