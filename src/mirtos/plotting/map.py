from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_map(
        data_map: np.ndarray,
        count_map: np.ndarray,
        projection: WCS,
        *,
        title: Optional[str] = None,
        cmap: str = "inferno",
        percentile_clip: tuple[float, float] = (2, 98),
        colorbar_label: Optional[str] = None,
        savepath: Optional[Path] = None,
        dpi: int = 300,
        plot_center: bool = False,
        show_hitmap: bool = False,
        hitmap_cmap: str = "magma",
        hitmap_log: bool = False,
        hitmap_title: str = "Hit map",
        marker_kwargs: Optional[dict] = None,
        label_kwargs: Optional[dict] = None,
        beam_fwhm_arcsec: Optional[float] = None,
        beam_center_world: Optional[tuple[float, float]] = None,
        beam_kwargs: Optional[dict] = None) -> tuple[plt.Figure, Union[plt.Axes, tuple[plt.Axes, plt.Axes]]]:

    mask = count_map > 0

    if np.any(mask):
        vmin, vmax = np.nanpercentile(data_map[mask], percentile_clip)
        ys, xs = np.where(mask)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    else:
        vmin, vmax = np.nanpercentile(data_map, percentile_clip)
        xmin, xmax = 0, data_map.shape[1] - 1
        ymin, ymax = 0, data_map.shape[0] - 1

    if show_hitmap:
        fig, (ax_map, ax_hits) = plt.subplots(
            1,
            2,
            figsize=(11, 5),
            tight_layout=True,
            subplot_kw={"projection": projection})
    else:
        fig, ax_map = plt.subplots(subplot_kw=dict(projection=projection))
        ax_hits = None

    im = ax_map.imshow(
        data_map,
        cmap=cmap,
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax)

    ax_map.set_xlim(xmin - 0.5, xmax + 0.5)
    ax_map.set_ylim(ymin - 0.5, ymax + 0.5)

    if title is not None:
        ax_map.set_title(title)

    cbar = fig.colorbar(im, ax=ax_map, shrink=0.6, pad=0.03)
    if colorbar_label is not None:
        cbar.set_label(colorbar_label)

    mk = {"marker": "x", "s": 60}
    if marker_kwargs:
        mk.update(marker_kwargs)

    lk = {"fontsize": 9, "ha": "left", "va": "bottom"}
    if label_kwargs:
        lk.update(label_kwargs)


    if beam_fwhm_arcsec is not None:
        if beam_center_world is None:
            ra0, dec0 = projection.wcs.crval[0], projection.wcs.crval[1]
        else:
            ra0, dec0 = beam_center_world

        x0, y0 = projection.wcs_world2pix([[ra0, dec0]], 0)[0]

        r_deg = (beam_fwhm_arcsec / 2.0) / 3600.0
        cdelt = projection.wcs.cdelt
        px_scale_deg = np.mean(np.abs(cdelt[:2]))
        r_pix = r_deg / px_scale_deg

        # dizionario per plottare il beam
        bk = {"fill": False, "linewidth": 1.5, "alpha": 0.9, "color": "white"}
        if beam_kwargs:
            bk.update(beam_kwargs)

        # cerchio del beam
        ax_map.add_patch(Circle((x0, y0), r_pix, **bk))

    # plot della count_map
    if show_hitmap and ax_hits is not None:
        # lo1p: conteggi di ogni bin preciso per numeri piccoli
        hits_img = np.log1p(count_map) if hitmap_log else count_map

        im_h = ax_hits.imshow(
            hits_img,
            cmap=hitmap_cmap,
            origin="lower",
            interpolation="nearest")

        ax_hits.set_xlim(xmin - 0.5, xmax + 0.5)
        ax_hits.set_ylim(ymin - 0.5, ymax + 0.5)
        ax_hits.set_title(hitmap_title + (" (log1p)" if hitmap_log else ""))

        cbar_h = fig.colorbar(im_h, ax=ax_hits, shrink=0.6, pad=0.03)
        cbar_h.set_label("Hits" + (" (log1p)" if hitmap_log else ""))

    if plot_center:
        ax_map.scatter(0.0,
                       0.0,
                       transform=ax_map.get_transform("world"),
                       s=120, edgecolors="k", facecolors="none", linewidths=2)


    if savepath is not None:
        fig.savefig(savepath, dpi=dpi)

    return (fig, (ax_map, ax_hits)) if show_hitmap else (fig, ax_map)
