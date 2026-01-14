import numpy as np
from tqdm import tqdm
from scipy.stats import binned_statistic_2d
from astropy.wcs import WCS

#from mirtos.core.data_types import Subscan
#from mirtos.binner import Binner
from mirtos.cleaner_class import Cleaner



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


