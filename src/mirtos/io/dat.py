from pathlib import Path
import numpy as np

def to_dat(file_path: Path):
    ...

def read_dat(file_path: Path, comments: str = '#'):

    return np.genfromtxt(file_path, comments=comments)