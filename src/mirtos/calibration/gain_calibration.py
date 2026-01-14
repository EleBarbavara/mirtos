from skydipcalibration import SkydipCalibration
import numpy as np
import statistics
import scipy

def gain_from_hf_noise(num_feed, sample_freq, tsdt):
        
        hfn_feeds = []
    
        for i in range(0, num_feed): 
            '''
            #ps = (np.abs(np.fft.fft(ts[i]))**2)
            #freqs0 = np.fft.fftfreq(n=len(ts[i]), d=tstep)
            #freqs = freqs0[0:len(freqs0)//2]
            #ps = ps[0:len(freqs0)//2]
            #idx = np.argsort(freqs)
            '''
            freqs, ps = scipy.signal.periodogram(tsdt[i], fs=sample_freq, scaling='density')
            maskk = freqs > 60
            masked_freq = freqs[maskk]
            masked_ps = np.sqrt(ps[maskk])
            mean_noise = np.mean(masked_ps)
        
            hfn_feeds.append(mean_noise)
        
        hfn_feeds = np.array(hfn_feeds)
        mean_noise_tot = np.mean(hfn_feeds)
        #print("Mean high frequency noise = ", mean_noise_tot)

        return hfn_feeds / mean_noise_tot

def responsivity(cfg,   subscan):

    sample_freq = subscan.sample_freq
    num_feed = subscan.num_feed

    
    if type(cfg.paths.skydip)==str:
        #Calibrates with a skydip 
        skydip_cal = SkydipCalibration(skydip_fits=cfg.paths.skydip, tau_atm=cfg.paths.tau, T_atm=cfg.paths.T_atm)
        gain_list =  skydip_cal.resps
    
        #excludes channels not included in the beam map

        for i in subscan.excl_feed:
            if  len(gain_list)-1<i:
                count = 0
            else:
                gain_list = list(gain_list)
                gain_list.pop(i) 


    elif cfg.paths.skydip==False:
        
        gain_list = gain_from_hf_noise(num_feed, sample_freq, subscan.tod_raw)
        
    return gain_list

def calibrate_tod(subscan):
     


        return 