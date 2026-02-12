from typing import Optional

import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt


def plot_map(map_: np.ndarray,
             projection: WCS,
             fig: Optional[plt.Figure],
             vmin: float = None,
             vmax: float = None):

    if not fig:
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    else:
        ax = fig.axes

    ax.imshow(map_, cmap='viridis', origin="lower", interpolation="bilinear", vmin=vmin, vmax=vmax)

    return fig, ax