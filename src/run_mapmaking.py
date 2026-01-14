import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import sys
import glob

from mirtos.core.data_types import Subscan
#from mirtos.binner import Binner
#from mirtos.cleaner_class import Cleaner
from mirtos.core.config_types import load_config



def main():
    path = sys.argv[1]
    cfg = load_config(path)
    tods_dir = cfg.paths.tods 
    #TO DO: estrarre fits file da piu cartelle
    tods_files = glob.glob(os.path.join(tods_dir, "*.fits"))
    sorted(tods_files)
    print('Making map of ', tods_dir)
    print('Total number of subscan:', len(tods_files))
    
    subscan = Subscan()
    subscan.initialize(cfg, system=cfg.telescope)

    

if __name__==main():
    main()
