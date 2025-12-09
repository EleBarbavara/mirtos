from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

class SkydipCalibration:
    def __init__(self, skydip_fits, T_atm, tau_atm):
        
        self.skydip_fits = skydip_fits
        self.T_atm = T_atm
        self.tau_atm = tau_atm

        hdul = fits.open(self.skydip_fits)
        self.dt = hdul["DATA TABLE"].data
        self.pt = hdul["PH TABLE"].data

        nchan = len(self.pt[0])

        self.mask = self.dt["flag_track"].astype(bool)

        self.el_rad = self.dt["el"][self.mask]
        self.z = np.pi/2 - self.el_rad
        self.airmass = 1 / np.cos(self.z)

        self.T_skydip = self.sky_temp()

        resps = []
        
        self.channels = np.arange(nchan)
        resps = []

        for i in self.channels:

            resp = self.fit_skydip(i)
            resps.append(resp)

        self.resps = np.array(resps)        
        self.med_resp = np.median(self.resps)
        self.resps_norm = np.array(resps) / self.med_resp      

    def sky_temp(self):
        return self.T_atm * (1-np.exp(-self.tau_atm*self.airmass))
    
    def sky_temp_2ord(self):
        return self.T_atm * (1-np.exp(-self.tau_atm*self.airmass))
    
    def fit_skydip(self, channel):
        
        tod = self.pt["chp_"+str(channel).zfill(3)][self.mask]
        pars = np.polyfit(self.T_skydip, tod, 1)
        model = np.poly1d(pars)
        resp = pars[0]

        return resp
   
    def plot_hist(self):

        fig, (ax1,ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15,7), dpi=200)

        
        hst = ax1.hist(self.resps,bins=15)
        ax1.set_title("SKYDIP")
        ax1.set_xlabel("gain [rad/K]")
        ax1.set_ylabel("count")
        ax1.vlines(self.med_resp, ymin=min(hst[0]), ymax=max(hst[0])+10, color="k", ls="dashed", label="Median = "+str(round(self.med_resp,3))+" rad/K")
        ax1.set_ylim(0,max(hst[0])+10)
        ax1.legend()
        ax1.grid(alpha=0.5)

        ax2.scatter(self.channels, self.resps, color="k", marker="+")
        ax2.set_ylim(max(self.resps)*1.3, min(self.resps)*1.3)
        ax2.set_xlabel("channel")
        ax2.set_ylabel("responsivity [rad/K]")
        ax2.grid(alpha=0.5)
