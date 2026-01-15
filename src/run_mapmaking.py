import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
import sys
import glob

from mirtos.core.data_types import Subscan
from mirtos.mapmaking.mapmaking import Binner
import mirtos.calibration as calibration
from mirtos.filtering.filters import Cleaner
from mirtos.core.config_types import load_config

import logging

logger = logging.getLogger(__name__)

def main():

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='run_mapmaking_log.log',
                        filemode='w')
    
    console = logging.StreamHandler()     # define a Handler which writes INFO messages or higher to the sys.stderr
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s') # set a format which is simpler for console use
    console.setFormatter(formatter)# tell the handler to use this format
    logging.getLogger().addHandler(console)

    path = sys.argv[1]
    cfg = load_config(path)
    tods_dir = cfg.paths.tods 
    #TO DO: estrarre fits file da piu cartelle
    tods_files = glob.glob(os.path.join(tods_dir, "*.fits"))
    sorted(tods_files)
    logger.info('Making map of '+str(tods_dir))
    logger.info('Total number of subscan: ' + str(len(tods_files)))

    subscan = Subscan()
    subscan.initialize(cfg, system=cfg.telescope)
    
    name_target = cfg.name_target
    data_obs = cfg.date_obs
    bin_mode = cfg.binner.frame
    projection = cfg.binner.projection


    binner = Binner(bin_mode=bin_mode, projection=projection)

    logger.info("bin mode = "+binner.mode)
    logger.info("Projection = "+binner.projection)

    if type(cfg.paths.skydip) == str:
        resps = calibration.gain_calibration.responsivity(cfg, 0, 0, 0)
    
    cleaner = Cleaner(cfg)
    cleaner.init_cleaner()

    part = task.partial(task.enrich_dataframe)
    scan = task.load_data(tods_dir, part)

if __name__==main():
    main()
