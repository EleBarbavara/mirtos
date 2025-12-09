import yaml
from astropy.io import fits
import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import statistics
import scipy

from src.mirtos.calibration.skydipcalibration import SkydipCalibration


def get_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config_data

def check_number_channel(tods_files):
    num_feed_all = []
    for filename in tods_files:
        hdr = []
        hdul = fits.open(filename)
        hdr_all = hdul['PH TABLE'].header
        for i in range(8, len(hdr_all)-1, 2): #faccio len(header)-1 perche l'ultimo header non lo voglio. Se non metto il -1 me lo salva
            hdr.append(hdr_all[i])
        num_feed = int(len(hdr))
        num_feed_all.append(num_feed)
    for i in range(len(num_feed_all)):
        if num_feed_all[i]!= num_feed_all[0]:
            print("Number of detectors change during subscans.")
    
    return num_feed
            

def exclude_pixel(ts, num_feed, dati_beammap):
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
    ts = list(ts)
    count = 0
    
    for i in excl_feed:
        if  len(ts)-1<i:
            #print('No channel ', i)
            count +=1
        else:
            ts.pop(i) # = np.delete(ts, i, axis=0)
        #pixel_mask.pop(i) # = np.delete(pixel_mask, i, axis=0)
    num_feed -= (len(excl_feed)-count)
    print(np.shape(offset_x), num_feed, excl_feed)
    
    return offset_x, offset_y, ts, num_feed, excl_feed

def lin_func(x, m, q):
    return m*x+q

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def lindetrend(x, time, mode='cutted'):
    '''
    possible mode to detrend the timestream:
    -'none' = detrend not applied
    - 'sigma' = perform a sigma clip around the source and makes a linear fit of the uncutted metadata
    - 'cutted' = cutting the firsts and lasts 300 elements of the timestream and makes a linear fit on this metadata
    '''
    
    if mode == 'none':
        res=np.copy(x)
        pixel_mask = np.full(int(len(x)), False)
    
    if mode == 'cutted':
        '''
        last_slice = int(len(x)/16)
        first_slice = int(len(x)/4)
        first_mask = np.full(first_slice, True)
        last_mask = np.full(last_slice, True)
        second_mask = np.full(int(len(x))-first_slice-last_slice, False)
        tot_mask = np.hstack((first_mask, second_mask, last_mask))
        '''
        pixel_mask = []
        #seleziono solo il primo e l'ultimo secondo di acquisizione + qualcosina
        slice = int(len(x)*0.1)
        #print(slice)
        idx1 = np.copy(slice)
        idx2 = int(len(x)) - slice
        tot_mask = np.ones(len(x)) 
        tot_mask[idx1:idx2] = 1 
        tot_mask = tot_mask.astype(bool)
        time = np.array(time)
        xdata = time[tot_mask]
        ydata = x[tot_mask]
        parfit, covfit = curve_fit(f=lin_func, xdata=xdata, ydata=ydata)
        res = x - lin_func(time, *parfit)
        pixel_mask = np.copy(np.array(np.array(tot_mask)))
    
    if mode == 'sigma':
        pixel_mask = []
        filtered_data = sigma_clip(x, sigma=6, maxiters=10) #result: masked array = masked array + mask (true = wanted values, false = masked values)
        mask = np.logical_not( np.ma.array(filtered_data).recordmask )
        pixel_mask = np.copy(mask)
        filt = x[mask]
        time = np.array(time)
        timestep = time[mask]
        #linear fit and subtraction of the best fit from the metadata (detrend)
        parfit, covfit = curve_fit(f=lin_func, xdata=timestep, ydata=filt)
        res = x - lin_func(time, *parfit)
        
    return res, pixel_mask

def remove_baseline(subscan_table, subscan_masked, time, mode='masked'):
    '''
    possible mode to detrend the timestream:
    -'none' = detrend not applied
    - 'sigma' = perform a sigma clip around the source and makes a linear fit of the uncutted metadata
    - 'cutted' = cutting the firsts and lasts 300 elements of the timestream and makes a linear fit on this metadata
    '''
    
    x = subscan_table['tod_raw']
    time = range(len(x))
    x_masked = subscan_masked['tod_raw']
    time_masked = range(len(x_masked))
    
    #linear fit and subtraction of the best fit from the metadata (detrend)
    parfit = np.polyfit(time_masked, x_masked, deg=4) #parfit, covfit = curve_fit(f=lin_func, xdata=time_masked, ydata=x_masked)
    polfit = np.poly1d(parfit)
    res = x - polfit(time) #lin_func(time, *parfit)
    
    return res

def responsivity(cfg, sample_freq, num_feed, tsdt):
    if type(cfg['paths']['skydip'])==str:
        skydip_cal = SkydipCalibration(skydip_fits=cfg['paths']['skydip'], tau_atm=cfg['paths']['tau'], T_atm=cfg['paths']['T_atm'])
        #ts = skydip_cal.med_resp*np.array(tsdt)
        return skydip_cal.resps
    
    elif cfg['paths']['skydip']==False:
        psd_feed = []
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
            mean_noise = statistics.mean(masked_ps)
        
            hfn_feeds.append(mean_noise)
        
        hfn_feeds = np.array(hfn_feeds)
        mean_noise_tot = statistics.mean(hfn_feeds)
        #print("Mean high frequency noise = ", mean_noise_tot)
        
        ts = tsdt.copy()
        for i in range(0, num_feed): 
            ts[i] /= hfn_feeds[i] 
            ts[i] *= mean_noise
        
        return ts

@np.vectorize
def rot(x,y,theta):
    '''
    xy = (x,y)

    mat_rot = ([np.cos(theta), -np.sin(theta)],
            [np.sin(theta),np.cos(theta)])
    
    ra_f, dec_f = np.matmul(mat_rot,xy)
    '''
    ra_f = x*np.cos(theta) - y*np.sin(theta)
    dec_f = x*np.sin(theta) + y*np.cos(theta)

    return ra_f, dec_f

def plot_map(mapp, cfg, namefile='', name='', save_map=False, wcs=None, beam=None, halo=None, vmax=None): 
    if wcs==None:
        #image = np.ones(num_timestep)
        fig, ax = plt.subplots()
        if vmax==None:
            im = ax.imshow(mapp, cmap='viridis', origin='lower') 
        else:
            im = ax.imshow(mapp, cmap='viridis', origin='lower', vmax=vmax)
        if beam!= None:
            beam[0].plot(color='white')
            ax.text(beam[1][0], beam[1][1]+6, f'beam \n 12"', c='white', fontsize='small', horizontalalignment='center')
        if halo!= None:
            halo.plot(color='white')
            #ax.text(beam[1][0], beam[1][1]+6, f'beam \n 12"', c='white', fontsize='small', horizontalalignment='center')
        
        #ax.set_title(f'{name}')
        ax.set_xlabel('RA [pixel]')
        ax.set_ylabel('Dec [pixel]')
        cbar = plt.colorbar(im) #, cax = cbaxes)
        if type(cfg['paths']['skydip'])==str:
                cbar_label='Kelvin [K]'
        else:
                cbar_label='Phase[rad]'
        cbar.set_label(label=cbar_label, size=16)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(axis='both', labelsize=16)
        plt.show()
        #plt.savefig(namefile)
        #plt.close()
        return ax
    else:
        #image = np.ones(num_timestep)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=wcs)
        if vmax==None:
            im = plt.imshow(mapp, cmap='viridis', origin='lower') 
        else:
            im = plt.imshow(mapp, cmap='viridis', origin='lower', vmax=vmax)
        if beam!= None:
            beam[0].plot(color='white')
            ax.text(beam[1][0], beam[1][1]+6, f'beam \n 12"', c='white', fontsize='small', horizontalalignment='center')
        if halo!= None:
            halo.plot(color='white')
        #ax.set_title(f'{str(name)}')
        ax.set_xlabel('RA', fontsize=18, color='black')
        ax.set_ylabel('Dec', fontsize=18, color='black')
        #cbaxes = fig.add_axes([0.85, 0.12, 0.03, 0.75])
        cbar = plt.colorbar(im) #, cax = cbaxes)
        if type(cfg['paths']['skydip'])==str:
                cbar_label='Kelvin [K]'
        else:
                cbar_label='Phase[rad]'
        cbar.set_label(label=cbar_label, size=16)
        cbar.ax.tick_params(labelsize=16)
        ax.tick_params(axis='both', labelsize=16)
        plt.show()
        #plt.savefig(namefile)
        #plt.close()
        return ax