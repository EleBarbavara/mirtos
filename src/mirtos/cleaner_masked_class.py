import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import scipy 
import statsmodels.api as sm
import random


def compute_common_mode(subscan_masked_table, num_feed, plot=False):
    #create the clipped timestreams (nan at each clipped element in the timestream)
    ts_masked = []
    for ch in range(num_feed):
        ts_masked.append(subscan_masked_table['tod_rb'][subscan_masked_table['ch']==ch]) #tod_dt
    
    common_mode = np.nanmean(ts_masked, axis=0)
    
    if plot == True:
        #print('3/3 : Plotting the common mode.....')
        plt.plot(ts_masked[0])
        plt.plot(common_mode)
        plt.title('Comparison of Channel 000 timestream and the signal of the general common mode')
        plt.xlabel('Timestep')
        plt.ylabel('Phase [rad]')
        plt.show()
        
    return common_mode

def corr_matrix(subscan_masked_table, num_feed, plot=False):
    ts_masked = []
    for ch in range(num_feed):
        ts_masked.append(subscan_masked_table['tod_dt'][subscan_masked_table['ch']==ch])
        
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

def custom_cm(subscan_table, num_feed, corr_matrix, mask, plot=False):
    ts = []
    for ch in range(num_feed):
        ts.append(subscan_table['tod_rb'][subscan_table['ch']==ch]) #tod_dt
    ts_masked = []
    subscan_masked = np.copy(subscan_table)
    for i in range(len(subscan_table)):
        if mask[i] is True:
            subscan_masked['tod_raw'] = np.nan
    for ch in range(num_feed):
        ts_masked.append(subscan_masked['tod_rb'][subscan_masked['ch']==ch]) #tod_dt
    
    
    '''
    common_mode_ch = []

    masked_ts = np.ma.array(ts, mask=np.invert(pixel_mask))
    source_masked = np.ma.MaskedArray(masked_ts, mask=np.isnan(masked_ts))
    #print('DIMENSIONE CORR MATRIX: ', np.shape(weight))
    '''
    common_mode_ch = []
    weight = np.copy(corr_matrix)
    source_masked = np.ma.MaskedArray(ts_masked, mask=np.isnan(ts_masked))
    for i in range(0, num_feed):
        weight[i][i] = 0
        common_mode_ch.append(np.ma.average(source_masked, weights=weight[i], axis=0))
    
    '''
    plt.plot(ts[0], label='Timestream')
    plt.plot(ts[0]-common_mode_ch[0], label = 'Timestram-common mode', c='red')
    #plt.plot(common_mode_ch[0], label='cm')
    #plt.title('Channel 000 ts - its individual common mode')
    plt.xlabel('Timestep')
    plt.ylabel('Phase [rad]')
    plt.legend()
    plt.show()'''
    
    subscan_table['cm_ch'] = np.hstack(common_mode_ch)
    subscan_masked = np.copy(subscan_table)
    for i in range(len(subscan_table)):
        if mask[i] is True:
            subscan_masked['tod_raw'] = np.nan
            subscan_masked['cm_ch'] = np.nan
            
    model = []
    b1 = []
    #b0 = []
    masked_cm_ch = []
    for ch in range(num_feed):
        masked_cm_ch.append(subscan_masked['cm_ch'][subscan_masked['ch']==ch])
    
    for i in range(0, num_feed):
        model = sm.OLS(ts_masked[i], masked_cm_ch[i]).fit()
        #print(model.summary())
        b1.append(model.params[0]) #slope of linear regression -> costante moltiplicativa da mettere davanti al common mode
        #b0 = model.params[1] #intercept of linear regression
    '''
    ts_raw_000 = ts_masked[5]
    ts_filtered_000 = ts[5]-b1[5]*common_mode_ch[5]

    f, P= scipy.signal.periodogram(ts_raw_000, fs=244.14, scaling='density')
    f1, P1= scipy.signal.periodogram(ts_filtered_000, fs=244.14, scaling='density')
    
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
    plt.savefig('./img/plot'+str(random.randint(0,500))+'.png')

    
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
        print('3/3 : Plotting an example of timestrea and PSD before and after removing the customized common mode.......')
        ts_raw_000 = ts_masked[0]
        ts_filtered_000 = ts[0]-b1[0]*common_mode_ch[0]

        f, P= scipy.signal.periodogram(ts_raw_000, fs=244.14, scaling='density')
        f1, P1= scipy.signal.periodogram(ts_filtered_000, fs=244.14, scaling='density')
        
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
    return common_mode_ch, b1, ts_masked

class Cleaner_masked():
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
        
        if all(mode == True for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            self.cm_type = 'not_filtered'
            
        elif self.cfg['filtering']['gen_cm'] == True:
            self.cm_type =  'gen_cm'
                
        elif self.cfg['filtering']['cust_cm'] == True:
            self.cm_type = 'cust_cm'
        
    def init_cleaner(self):    
        if all(mode == True for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            print('-----> You choose to not filter the metadata, are you sure? :)')
            self.cm_type = 'not_filtered'
            
        elif self.cfg['filtering']['gen_cm'] == True:
            print('Filtering the metadata with the general common mode.')
            self.cm_type =  'gen_cm'
                
        elif self.cfg['filtering']['cust_cm'] == True:
            print('Filtering the metadata with the customized common mode.')
            self.cm_type = 'cust_cm'
        
    def filter(self, subscan_table, mask, num_feed):
        if self.cfg['filtering']['use_detrend_tods']==True:
            ts = []
            for ch in range(num_feed):
                ts.append(subscan_table['tod_dt'][subscan_table['ch']==ch])
        else:
            ts = []
            for ch in range(num_feed):
                ts.append(subscan_table['tod_raw'][subscan_table['ch']==ch])
        
        if all(mode == True for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            raise ValueError('Choose one method to filter the metadata: either with the general common mode or the custom common mode.')
        
        elif all(mode == False for mode in [self.cfg['filtering']['gen_cm'], self.cfg['filtering']['cust_cm']]):
            return ts, [0]*len(ts[0])*num_feed
            
        elif self.cfg['filtering']['gen_cm'] == True:
            subscan_masked = np.copy(subscan_table)
            for i in range(len(subscan_table)):
                if mask[i] is True:
                    subscan_masked['tod_raw'] = np.nan
            cm = compute_common_mode(subscan_masked, num_feed, plot=self.cfg['filtering']['plot_cm'])
            self.cm = cm
            
            ts_filt = []
            for i in range(num_feed):
                ts_filt.append(ts[i] - cm)
            
            cms = [cm]*num_feed
            return ts_filt, cms
                
        
        elif self.cfg['filtering']['cust_cm'] == True:
            subscan_masked = np.copy(subscan_table)
            for i in range(len(subscan_table)):
                if mask[i] is True:
                    subscan_masked['tod_raw'] = np.nan
            
            corr_matrix_dt = corr_matrix(subscan_masked, num_feed, plot=self.cfg['filtering']['plot_corr_matrix']) 

            self.cust_common_mode, self.b1, self.source_masked = custom_cm(subscan_table, num_feed, corr_matrix_dt, mask,
                                            plot=self.cfg['filtering']['plt_cust_cm'])
            
            ts_filt = []
            cms = []
            for i in range(num_feed):
                ts_filt.append(ts[i] - self.b1[i]*self.cust_common_mode[i])
                cms.append(self.b1[i]*self.cust_common_mode[i]) 

            
            
            self.cms = cms
            subscan_table['tod_filt'] = np.hstack(ts_filt)
            subscan_masked = subscan_table[~mask]
            self.corr_matrix_cust = corr_matrix(subscan_masked, num_feed, plot=self.cfg['filtering']['plot_corr_matrix'])
            
            return ts_filt, cms
        
        else:
            raise ValueError('What are you doing with the configs file? Check the filtering section.')
    
        
        