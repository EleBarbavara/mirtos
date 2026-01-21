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

    # problema: chiuederebbe in automatico il fits
    with fits.open(filename) as hdul:
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