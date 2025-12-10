from multiprocessing import Pool
from pandas import DataFrame
from pathlib import Path
from itertools import repeat
from scipy.optimize import curve_fit
from scipy import signal

import os
import numpy as np
from astropy.table import Table, vstack

import lib
from subscan_class import Subscan
from binner import Binner
from cleaner_class import Cleaner
from cleaner_masked_class import Cleaner_masked

cfg = lib.get_config_file("../../configs/config.yaml")

tods_dir = cfg['paths']['tods']
tods_names = os.listdir(tods_dir)
tods_files = []
for name in tods_names:
    tods_files.append(os.path.join(tods_dir, name))
tods_files.sort()

subscan = Subscan(cfg)

name_target = cfg['name_target']
data_obs = cfg['date_obs']

bin_mode = cfg['binner']['frame']
projection = cfg['binner']['projection']
binner = Binner(bin_mode=bin_mode, projection=projection)

if type(cfg['paths']['skydip']) == str:
    resps = lib.responsivity(cfg, 0, 0, 0)
    
def lin_func(x, m, q):
    return m*x+q
    
def band_pass_filter(time_series, cuton_freq, cutoff_freq, sampling_rate, order=4):
    """
    Apply a band-pass Butterworth filter to a time series.

    Parameters:
    - time_series (array-like): The input time-series metadata.
    - cuton_freq (float): The lower cutoff frequency (cut-on frequency) in Hz.
    - cutoff_freq (float): The upper cutoff frequency (cut-off frequency) in Hz.
    - sampling_rate (float): The sampling rate of the metadata in Hz.
    - order (int): The order of the Butterworth filter (default is 4).

    Returns:
    - filtered_series (numpy array): The filtered time-series metadata.
    """
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cuton = cuton_freq / nyquist  # Normalized cut-on frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cut-off frequency
    
    # Design Butterworth band-pass filter
    b, a = signal.butter(order, [normal_cuton, normal_cutoff], btype='band', analog=False)
    
    # Apply the band-pass filter
    filtered_series = signal.filtfilt(b, a, time_series)
    
    return filtered_series

def low_pass_filter(time_series, cutoff_freq, sampling_rate, order=4):
    """
    Apply a low-pass Butterworth filter to a time series.

    Parameters:
    - time_series (array-like): The input time-series metadata.
    - cutoff_freq (float): The cutoff frequency of the low-pass filter in Hz.
    - sampling_rate (float): The sampling rate of the metadata in Hz.
    - order (int): The order of the Butterworth filter (default is 4).

    Returns:
    - filtered_series (numpy array): The filtered time-series metadata.
    """
    nyquist = 0.5 * sampling_rate  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    
    # Design Butterworth low-pass filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    filtered_series = signal.filtfilt(b, a, time_series)
    
    return filtered_series

def create_scan(filename):
    n_subscan = str(filename[0]).split('.fits')[0].split('_')[-1]
    #nsubscans.append(n_subscan)
    #print(filename)
    subscan.extract_data(str(filename[0]), cfg)
    
    #skippo i file vuoti
    if any(len(lst) == 0 for lst in [subscan.par_angle, subscan.ra_scan, subscan.dec_scan, subscan.timestream_raw]) is True:
        print('Empty file skipped: ', str(filename[0]))
        cleaner = Cleaner(cfg)
        subscan_table = Table([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], ['bad']],
                            names=['n_subscan', 'ch', 't', 'lon', 'lat', 'tod_raw', 'tod_dt', 'tod_filt', 'mask', cleaner.cm_type, 'good/bad'],
                            dtype=('S3', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'b', 'f8', 'S4'))
        return subscan_table

    
    elif all(len(lst) != subscan.num_timestep for lst in [subscan.par_angle, subscan.ra_scan, 
                                                          subscan.dec_scan]):
        raise ValueError(f'The lengths of lists do not match. Length timstep = {subscan.num_timestep}. Length par_angle = {len(subscan.par_angle)}. Length RA = {len(subscan.ra_scan)}. Length dec = {len(subscan.dec_scan)}.')
    
    else:
        #channel_table = Table([subscan.offset_x, subscan.offset_y], names=['xOffset', 'yOffset'])
        
        ch_list = [int(0)]*subscan.num_timestep
        for i in range(1, subscan.num_feed):
            ch_list += [int(i)]*subscan.num_timestep
            
        x, y = binner.proj_radec_to_xy(ra=subscan.ra_scan, dec=subscan.dec_scan, 
                                ra0=subscan.ra_center, dec0=subscan.dec_center, projection=projection)
        binner.conv_xy_to_latlon(x=x, y=y, par_angle=subscan.par_angle, num_feed=subscan.num_feed,
                            offset_x = np.array(subscan.offset_x), offset_y=np.array(subscan.offset_y),
                            center_ra=subscan.ra_center, center_dec=subscan.dec_center)
        
        radius = np.deg2rad(cfg['filtering']['radius']/3600)
        
        if cfg['paths']['skydip'] == False:
            #ts_raw = lib.responsivity(cfg, subscan.sample_freq, subscan.num_feed, subscan.timestream_raw)
            ts_raw = subscan.timestream_raw
            
            
        elif type(cfg['paths']['skydip']) == str:
            ts_raw = []
            resps_copy = np.copy(resps)
            for i in subscan.excl_feed:
                if  len(resps_copy)-1<i:
                    count = 0
                else:
                    resps_copy = list(resps_copy)
                    resps_copy.pop(i) 
            
            np.save('../../metadata/gain.npy', np.array(resps_copy))
            
            for i in range(subscan.num_feed):
                #TO DO associare i giusti file/dati per la calibrazione
                #TO DO calibrazione con piu calibratori? 
                z = np.pi/2 - np.array(subscan.el_scan + subscan.offset_y_single[i])
                ts_raw.append(subscan.timestream_raw[i]/(resps_copy[i]*np.exp(-cfg['paths']['tau']/np.cos(z))))
        
        if radius==0 or radius==False:
            cleaner = Cleaner(cfg)
            
            tsdt = []
            pixel_mask = []
            
            for i in range(subscan.num_feed): 
                dt_ts, maskfeed = lib.lindetrend(ts_raw[i], subscan.range_timestep, mode=cfg['filtering']['baseline_rem'])
                tsdt.append(dt_ts)
                pixel_mask.append(maskfeed)
            
            ts_filt, cm = cleaner.filter(subscan, tsdt, pixel_mask) 
            pixel_mask = np.hstack(pixel_mask)
            
        else:
            subscan_table_raw = Table([[0], [0], [0], [0]], 
                                names=['ch', 'lon', 'lat', 'tod_raw'],
                                dtype=('i4', 'f8', 'f8', 'f8'))
            
            cleaner = Cleaner_masked(cfg) 
            
            subscan_table_raw = Table([ch_list, list(np.hstack(binner.lon)), list(np.hstack(binner.lat)), np.hstack(ts_raw)], 
                                names=['ch', 'lon', 'lat', 'tod_raw'],
                                dtype=('i4', 'f8', 'f8', 'f8'))
            
            dist_from_center = np.sqrt((subscan_table_raw['lon'] - subscan.ra_center)**2 + (subscan_table_raw['lat']-subscan.dec_center)**2)
            mask = dist_from_center <= radius
            subscan_masked = np.copy(subscan_table_raw)
            #subscan_masked = subscan_masked[~mask]
            
            for i in range(len(subscan_table_raw)):
                if mask[i] is True:
                    subscan_masked['tod_raw'] = np.nan
            
            tsdt_rb = []
            for ch in range(subscan.num_feed): 
                tsdt_rb.append(lib.remove_baseline(subscan_table_raw[subscan_table_raw['ch'] == ch], subscan_masked[subscan_masked['ch'] == ch], subscan.range_timestep, mode='masked'))
            subscan_table_raw = Table([ch_list, list(np.hstack(binner.lon)), list(np.hstack(binner.lat)), np.hstack(ts_raw), np.hstack(tsdt_rb), mask], 
                                names=['ch', 'lon', 'lat', 'tod_raw', 'tod_rb', 'mask'],
                                dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'b'))
            
            
            for n_dt in range(4):
                tsdt = []
                for i in range(subscan.num_feed):
                    if n_dt == 0:
                        tod = subscan_table_raw[subscan_table_raw['ch']==i]['tod_rb']
                    else:
                        tod = ts[i]
                    maski = subscan_table_raw[subscan_table_raw['ch']==i]['mask']
                    time = range(len(tod[maski]))
                    parfit, covfit = curve_fit(f=lin_func, xdata=time, ydata=tod[maski])
                    tsdt.append(tod - lin_func(range(len(tod)), *parfit))
                ts = np.copy(tsdt)
                
                
            subscan_table_raw = Table([ch_list, list(np.hstack(binner.lon)), list(np.hstack(binner.lat)), np.hstack(ts_raw), np.hstack(tsdt_rb), mask, np.hstack(tsdt)], 
                                names=['ch', 'lon', 'lat', 'tod_raw', 'tod_rb', 'mask', 'tod_dt'],
                                dtype=('i4', 'f8', 'f8', 'f8', 'f8', 'b', 'f8'))
            
            ts_filt, cm = cleaner.filter(subscan_table_raw, mask, subscan.num_feed)
            pixel_mask = mask
            
        #ts_bp = [low_pass_filter(tod, cutoff_freq=30, sampling_rate=244.14) for tod in ts_filt]
        
        qual_subscans = []
        for i in range(subscan.num_feed):
            if np.nanmax(subscan.timestream_raw[i]) > np.mean(subscan.timestream_raw[i]) + 7*np.std(subscan.timestream_raw[i]) or np.nanmin(subscan.timestream_raw[i]) < np.mean(subscan.timestream_raw[i]) - 7*np.std(subscan.timestream_raw[i]):   
                qual_subscans.append('bad')
            else:
                qual_subscans.append('good')
        
        if all(s =='bad' for s in qual_subscans):
            bad_subscans = 'bad'
        else:
            bad_subscans = 'good'
            
        ss = [n_subscan]*subscan.num_timestep*subscan.num_feed
        t_list = list(subscan.time)*subscan.num_feed
        subscan_table = Table([ss, ch_list, t_list, list(np.hstack(binner.lon)), list(np.hstack(binner.lat)), list(subscan.az_scan)*subscan.num_feed, 
                               list(subscan.el_scan)*subscan.num_feed, np.hstack(ts_raw), np.hstack(tsdt), 
                            np.hstack(ts_filt), pixel_mask, np.hstack(cm), [bad_subscans]*subscan.num_timestep*subscan.num_feed],
                            names=['n_subscan', 'ch', 't', 'lon', 'lat', 'az', 'el', 'tod_raw', 'tod_dt', 'tod_filt', 'mask', cleaner.cm_type, 'good/bad'],
                            dtype=('S3', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'b', 'f8', 'S4'))
        
        return subscan_table
    

def enrich_dataframe(data: DataFrame):
    data = data
    
def get_symbol_for_path(path: Path, enrich_function: type[enrich_dataframe]  = lambda x: x)-> DataFrame:
    """
    Get dataframe from path
    """
    subscan_table = create_scan(path)
    enriched = enrich_function(subscan_table)
    return enriched

def load_data(root_path: str, enrich_function: type[enrich_dataframe])-> DataFrame:
    files = [Path(root_path, i) for i in os.listdir(root_path) if i.endswith('.fits')]
    files.sort()
    #files=[files[90]]
    #print(files)
    print(f"Loaded {len(files)} files")
    with Pool() as pool:
        datasets = pool.map(get_symbol_for_path, zip(files, repeat(enrich_function)))
        return vstack(datasets)


