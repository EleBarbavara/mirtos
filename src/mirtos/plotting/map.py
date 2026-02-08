from typing import Optional

import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt


def plot_map(map_: np.ndarray, projection: WCS, fig: Optional[plt.Figure]):

    if not fig:
        fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    else:
        ax = fig.axes

    ax.imshow(map_, cmap='viridis', origin='lower')

    return fig, ax