import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy 
import statsmodels.api as sm #sm = silvia masi
import logging

def compute_common_mode(ts, pixel_mask, plot=False):
    #create the clipped timestreams (nan at each clipped element in the timestream)
        noise_ts = np.copy(ts)
        
        #logging.info('1/3 : Compunting the timestream on which compute the common mode.....')
        for tod, mask in zip(noise_ts, pixel_mask):
            tod[np.logical_not(mask)]=np.nan
        
        #fcompute the average -> common mode
        #logging.info('2/3 : Compunting the common mode.....')
        common_mode = np.nanmean(noise_ts, axis=0)
        
        if plot == True:
            #logging.info('3/3 : Plotting the common mode.....')
            plt.plot(ts[0])
            plt.plot(common_mode)
            plt.title('Comparison of Channel 000 timestream and the signal of the general common mode')
            plt.xlabel('Timestep')
            plt.ylabel('Phase [rad]')
            plt.show()
            
        return common_mode

def custom_cm(ts, num_feed, corr_matrix, pixel_mask, plot=False):
    common_mode_ch = []
    weight = np.copy(corr_matrix)
    
    masked_ts = np.ma.array(ts, mask=np.invert(pixel_mask))
    source_masked = np.ma.MaskedArray(masked_ts, mask=np.isnan(masked_ts))
    #logging.info('DIMENSIONE CORR MATRIX: ', np.shape(weight))
    for i in range(0, num_feed):
        weight[i][i] = 0
        common_mode_ch.append(np.average(source_masked, weights=weight[i], axis=0))
    
    '''
    plt.plot(ts[0], label='Timestream')
    plt.plot(ts[0]-common_mode_ch[0], label = 'Timestram-common mode', c='red')
    #plt.plot(common_mode_ch[0], label='cm')
    #plt.title('Channel 000 ts - its individual common mode')
    plt.xlabel('Timestep')
    plt.ylabel('Phase [rad]')
    plt.legend()
    plt.show()'''
    
    model = []
    b1 = []
    #b0 = []
    masked_cm_ch = np.ma.array(common_mode_ch, mask=np.invert(pixel_mask))
    for i in range(0, num_feed):
        model = sm.OLS(masked_ts[i], masked_cm_ch[i]).fit()
        #logging.info(model.summary())
        b1.append(model.params[0]) #slope of linear regression -> costante moltiplicativa da mettere davanti al common mode
        #b0 = model.params[1] #intercept of linear regression
    '''
    plt.hist(b1, bins=150, label='$b_1$ from linear regression')
    plt.axvline(statistics.mean(b1), color='red', label='Mean value of $b_1$')
    #plt.title('Histogram of b1 constant from linear regression')
    plt.xlabel('$b_1$')
    plt.ylabel('N')
    plt.legend()
    plt.show()

    plt.hist(b0, bins=150, label='$b_0$ from linear regression')
    plt.axvline(statistics.mean(b1), color='red', label='Mean value of $b_0$')
    #plt.title('Histogram of b1 constant from linear regression')
    plt.xlabel('$b_0$')
    plt.ylabel('N')
    plt.legend()
    plt.show()

    plt.plot(ts[0], label='Original timestream')
    plt.plot(ts[0]-b1[0]*common_mode_ch[0], label = 'Filtered timestream', c='red')
    plt.xlabel('Timestep')
    plt.ylabel('Phase [rad]')
    plt.legend()
    plt.show()'''
    
    if plot==True:
        logging.info('3/3 : Plotting an example of timestrea and PSD before and after removing the customized common mode.......')
        ts_raw_000 = ts[0]
        ts_filtered_000 = ts[0]-b1[0]*common_mode_ch[0]

        f, P= scipy.signal.periodogram(ts_raw_000[pixel_mask[0]], fs=244.14, scaling='density')
        f1, P1= scipy.signal.periodogram(ts_filtered_000[pixel_mask[0]], fs=244.14, scaling='density')
        
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(13,5))
        ax1.plot(ts_raw_000, label='Original timestream')
        ax1.plot(ts_filtered_000, label = 'Filtered timestream', c='red')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Phase [rad]')
        ax1.legend()

        ax2.plot(f[1:], np.sqrt(P[1:]), label='Original timestream')
        ax2.plot(f1[1:], np.sqrt(P1[1:]), label = 'Filtered timestream', c='red')
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_ylabel(r'PSD [rad/$\sqrt{Hz}$]')
        ax2.set_xlabel('Frequency [Hz]')
        #plt.title('Channel 000 filtered power spectal density')
        ax2.legend()
        plt.show()
    
    #casting a np.ndarray
    common_mode_ch = np.array(np.array(common_mode_ch))
    b1 = np.array(np.array(b1))
    return common_mode_ch, b1, source_masked

def corr_matrix(ts, pixel_mask, plot=False):
    ts_copy = np.copy(ts)
    masks = np.copy(np.array(np.array(pixel_mask)))
    ts_masked = np.copy(np.array(np.array(ts_copy)))
    for tod, mask in zip(ts_masked, masks):
        tod[np.logical_not(mask)]=np.nan
    
    ts_masked = pd.DataFrame(ts_masked).T

    corr_matrix = ts_masked.corr()
    
    if plot == True :
        fig, ax = plt.subplots()
        im = ax.imshow(corr_matrix, cmap='coolwarm')
        im.set_clim(-1, 1)
        cbar = ax.figure.colorbar(im, ax=ax)#, format='% .2f' #quando c'è coeff di corr sull'immagine
        plt.title('Channels correlation matrix (only detrended)')
        plt.xlabel('Channel')
        plt.ylabel('Channel')
        plt.show()
    
    return np.matrix(corr_matrix)


class Cleaner():
    '''
        This class refers to the metadata filteirng. Its methods compute the general and customized common mode for each feed in order to minimize the noise
        (removing the pulse tube noise). The correlation matrixes are shown in order to see the progress done during this analysis.
        Then, a map of the filtered metadata is shown + a map in the region of high coverage of the scan.
        '''
    def __init__(self, cfg):
        #definisco tutte le variabili che poi userò con la classe MapMaker
        self.cfg = cfg
        self.cm = []
        self.corr_matrix = []
        self.b1 = []
        self.cust_common_mode = []
        
        if all(mode == True for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            self.cm_type = 'not_filtered'
            
        elif self.cfg.filtering.gen_cm == True:
            self.cm_type =  'gen_cm'
                
        elif self.cfg.filtering.cust_cm == True:
            self.cm_type = 'cust_cm'
        
    def init_cleaner(self):    
        if all(mode == True for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            logging.info('-----> You choose to not filter the metadata, are you sure? :)')
            self.cm_type = 'not_filtered'
            
        elif self.cfg.filtering.gen_cm == True:
            logging.info('Filtering the metadata with the general common mode.')
            self.cm_type =  'gen_cm'
                
        elif self.cfg.filtering.cust_cm == True:
            logging.info('Filtering the metadata with the customized common mode.')
            self.cm_type = 'cust_cm'
        
    def filter(self, subscan, tsdt, pixel_mask):
        if self.cfg.filtering.use_detrend_tods==True:
            ts = np.copy(tsdt)
        else:
            ts = np.copy(subscan.timestream_raw)
        
        if all(mode == True for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg.filtering.gen_cm, self.cfg.filtering.cust_cm]):
            return ts, [0]*len(ts[0])*subscan.num_feed
            
        elif self.cfg.filtering.gen_cm == True:
            cm = compute_common_mode(ts, pixel_mask, plot=self.cfg.filtering.plot_cm)
            self.cm = cm
            
            ts_filt = []
            for i in range(subscan.num_feed):
                ts_filt.append(ts[i] - cm)
            
            cms = [cm]*subscan.num_feed
            return ts_filt, cms
                
        
        elif self.cfg.filtering.cust_cm == True:
            corr_matrix_dt = corr_matrix(ts, pixel_mask, plot=self.cfg.filtering.plot_corr_matrix) 

            self.cust_common_mode, self.b1, self.source_masked = custom_cm(ts, 
                                            subscan.num_feed, 
                                            corr_matrix_dt, 
                                            pixel_mask,
                                            plot=self.cfg.filtering.plt_cust_cm)
            
            ts_filt = []
            cms = []
            for i in range(subscan.num_feed):
                ts_filt.append(ts[i] - self.b1[i]*self.cust_common_mode[i])
                cms.append(self.b1[i]*self.cust_common_mode[i]) 

            self.cms = cms
            self.corr_matrix_cust = corr_matrix(ts=ts_filt, pixel_mask=pixel_mask, plot=self.cfg.filtering.plot_corr_matrix)
            
            return ts_filt, cms
        
        else:
            raise ValueError('What are you doing with the configs file? Check the filtering section.')

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

