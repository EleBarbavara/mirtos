import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import sys
import glob

from src.mirtos import lib, task
from src.mirtos.core.data_types import Subscan
from src.mirtos.binner import Binner
from src.mirtos.cleaner_class import Cleaner
from src.mirtos.core.config_types import load_config



def main(path: str):
    cfg = load_config(path)
    tods_dir = cfg.paths.tods 
    #TO DO: estrarre fits file da piu cartelle
    tods_files = glob.glob(os.path.join(tods_dir, "*.fits"))
    sorted(tods_files)
    print('Making map of ', tods_dir)
    print('Total number of subscan:', len(tods_files))
    
    subscan = Subscan()
    subscan.initialize(cfg, telescope=cfg['telescope'])
    subscan.init_telescope()

    

if __name__==main():
    path = sys.argv[1]
    main(path)
