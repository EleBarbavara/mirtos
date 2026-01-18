# codice per leggere i fits
# seguire esempio pandas
import os
import glob
import numpy as np
from astropy.io import fits


def to_fits(file_path: str):
    ...

# i config che usiamo servono qui o potrebbero servire in scan.py?
# ritorna solo l'hdul mentre  process_subscan_file ritorna solo le cose che servono dell'hdul
def load_subscan_fits(filename):

    with fits.open(filename) as hdul:
        return hdul


@classmethod
def read_discocs_fits(cls, filename, config):
    # deve fare quello che fa load_data
    hdul = fits.open(filename) 
            
    #cls.srp_tz = np.mean(hdul['SERVO TABLE'].metadata['SRP_TZ']) #mm
    #cls.srp_ty = np.mean(hdul['SERVO TABLE'].metadata['SRP_TY']) #mm
    #cls.srp_tx = np.mean(hdul['SERVO TABLE'].metadata['SRP_TX']) #mm
    cls.flag_track = hdul['DATA TABLE'].data['flag_track']
    
    if config['flag_track'] == True:
        flag_track = ft.astype(bool)
    else:
        flag_track = np.ones(len(ft)).astype(bool)
    
    ft = hdul['DATA TABLE'].data['flag_track']
    cls.flag_track = ft > 0.5
    
    try: 
        angle_offset = 0 #np.pi/2
        par_angle_nan = hdul['DATA TABLE'].data['par_angle']
        cls.par_angle = par_angle_nan[np.logical_not(np.isnan(par_angle_nan))] + angle_offset
        ra_scan_nan = hdul['DATA TABLE'].data['raj2000']
        cls.ra_scan = ra_scan_nan[np.logical_not(np.isnan(ra_scan_nan))]
        dec_scan_nan = hdul['DATA TABLE'].data['decj2000']
        cls.dec_scan = dec_scan_nan[np.logical_not(np.isnan(dec_scan_nan))]
        az_scan_nan = hdul['DATA TABLE'].data['az']
        cls.az_scan = az_scan_nan[np.logical_not(np.isnan(az_scan_nan))]
        el_scan_nan = hdul['DATA TABLE'].data['el']
        cls.el_scan = el_scan_nan[np.logical_not(np.isnan(el_scan_nan))]
        time_all = hdul["DATA TABLE"].data["time"]
        cls.time = time_all[np.logical_not(np.isnan(dec_scan_nan))]
        cls.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
        cls.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
        '''
        angle_offset = 0 #np.pi/2
        cls.par_angle = hdul['DATA TABLE'].metadata['par_angle'][flag_track] + angle_offset
        cls.ra_scan = hdul['DATA TABLE'].metadata['raj2000'][flag_track] 
        cls.dec_scan = hdul['DATA TABLE'].metadata['decj2000'][flag_track]
        cls.time = hdul["DATA TABLE"].metadata["time"][flag_track]
        cls.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
        cls.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
        '''
    except:
        angle_offset = 0#-np.pi/2
        par_angle_nan = hdul['DATA TABLE INTERP'].data['par_angle_interpolated']
        cls.par_angle = par_angle_nan[np.logical_not(np.isnan(par_angle_nan))] + angle_offset
        ra_scan_nan = hdul['DATA TABLE INTERP'].data['raj2000_interpolated']
        cls.ra_scan = ra_scan_nan[np.logical_not(np.isnan(ra_scan_nan))]
        dec_scan_nan = hdul['DATA TABLE INTERP'].data['decj2000_interpolated']
        cls.dec_scan = dec_scan_nan[np.logical_not(np.isnan(dec_scan_nan))]
        cls.ra_center = hdul['PRIMARY'].header['HIERARCH RightAscension']
        cls.dec_center = hdul['PRIMARY'].header['HIERARCH Declination']
    
    cls.range_timestep = range(0, len(cls.par_angle), 1) #(1222, 147704) daisy #(1209, 9647) 30s #where the fits file is not 'NULL' or not zero
    cls.all_range_timestep.append(cls.range_timestep)
    cls.num_timestep = int(len(cls.range_timestep))
    cls.sample_freq = 244.140625 #256e6/2**20 
    cls.tstep = 1/cls.sample_freq
    cls.nyqfreq = cls.sample_freq/2
    
    cls.num_feed = len(hdul["PH TABLE"].data[0])
    cls.num_feed_all.append(cls.num_feed)
    
    hdr = ["chp_"+str(i).zfill(3) for i in range(cls.num_feed)]
    
    cls.timestream_raw = []
    #cls.timestream_raw = hdul['PH TABLE'].metadata[hdr[0]][flag_track]
    for i in range(0, cls.num_feed):
        #try:
        #cls.timestream_raw.append(hdul['PH TABLE'].metadata[hdr[i]][flag_track])
        ts_nan = hdul['PH TABLE'].data[hdr[i]]
        cls.timestream_raw.append(ts_nan[np.logical_not(np.isnan(ra_scan_nan))])
        #except:
        #    cls.timestream_raw.append([np.nan]*cls.num_timestep)

    return hdul

def read_tod_fits(dir_path: str):
    """
    Metdodo che ritorna le tod. in un altro metodo creiamo lo scan in un'altra modulo
    Args:
        dir_path:

    Returns:

    """

    # lista di file.fits per ogni TOD
    files = glob.glob(os.path.join(dir_path, '*.fits'))
    fits_data = []

    for file in files:
        with fits.open(file) as hdul:
            fits_data.append(hdul[1].data)

def check_number_channel():
    pass