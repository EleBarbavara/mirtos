from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

MarkerWorld = tuple[float, float, str]
MarkerPixel = tuple[float, float, str]


def plot_map(
        data_map: np.ndarray,
        count_map: np.ndarray,
        projection: WCS,
        std_map: Optional[np.ndarray] = None,
        *,
        title: Optional[str] = None,
        cmap: str = "inferno",
        percentile_clip: tuple[float, float] = (2, 98),
        colorbar_label: Optional[str] = None,
        savepath: Optional[Path] = None,
        dpi: int = 300,
        hitmap_cmap: str = "magma",
        hitmap_log: bool = False,
        hitmap_title: str = "Hit map",
        markers_world: Optional[Sequence[MarkerWorld]] = None,
        markers_pix: Optional[Sequence[MarkerPixel]] = None,
        marker_kwargs: Optional[dict] = None,
        label_kwargs: Optional[dict] = None,
        beam_fwhm_arcsec: Optional[float] = None,
        beam_center_world: Optional[tuple[float, float]] = None,
        beam_kwargs: Optional[dict] = None) -> tuple[plt.Figure, Union[plt.Axes, tuple[plt.Axes, ...]]]:

    if count_map.shape != data_map.shape:
        raise ValueError("count_map must have the same shape as data_map")
    if std_map is not None and std_map.shape != data_map.shape:
        raise ValueError("std_map must have the same shape as data_map")

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

    n_panels = 3 if std_map is not None else 2
    figsize = (16, 5) if std_map is not None else (11, 5)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        tight_layout=True,
        subplot_kw={"projection": projection})

    if n_panels == 2:
        ax_map, ax_hits = axes
        ax_std = None
    else:
        ax_map, ax_hits, ax_std = axes

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

    if markers_pix:
        for x, y, lab in markers_pix:
            ax_map.scatter([x], [y], **mk)
            if lab:
                ax_map.text(x, y, f" {lab}", **lk)

    if markers_world:
        for ra_deg, dec_deg, lab in markers_world:
            x, y = projection.wcs_world2pix([[ra_deg, dec_deg]], 0)[0]
            ax_map.scatter([x], [y], **mk)
            if lab:
                ax_map.text(x, y, f" {lab}", **lk)

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

        bk = {"fill": False, "linewidth": 1.5, "alpha": 0.9, "color": "white"}
        if beam_kwargs:
            bk.update(beam_kwargs)

        ax_map.add_patch(Circle((x0, y0), r_pix, **bk))

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

    if ax_std is not None:
        std_mask = np.isfinite(std_map) & mask
        if np.any(std_mask):
            std_vmin, std_vmax = np.nanpercentile(std_map[std_mask], percentile_clip)
        else:
            std_vmin, std_vmax = np.nanpercentile(std_map, percentile_clip)

        im_std = ax_std.imshow(
            std_map,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            vmin=std_vmin,
            vmax=std_vmax)

        ax_std.set_xlim(xmin - 0.5, xmax + 0.5)
        ax_std.set_ylim(ymin - 0.5, ymax + 0.5)
        ax_std.set_title("STD map")

        cbar_std = fig.colorbar(im_std, ax=ax_std, shrink=0.6, pad=0.03)
        cbar_std.set_label("STD")


    if savepath is not None:
        fig.savefig(savepath, dpi=dpi)

    return fig, tuple(ax for ax in (ax_map, ax_hits, ax_std) if ax is not None)
