from pathlib import Path
from typing import Protocol

import numpy as np
from astropy.io import fits

# duck typing
class MapProductLike(Protocol):
    data_map: np.ndarray
    count_map: np.ndarray
    std_map: np.ndarray | None
    wcs: object
    meta: dict[str, str | float | int]


def to_fits(file_path: str | Path, product: MapProductLike) -> Path:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    primary_header = fits.Header()
    for key, value in getattr(product, "meta", {}).items():
        if value is None:
            continue
        primary_header[str(key).upper()[:8]] = value

    header_wcs = product.wcs.to_header()

    hdus = [
        fits.PrimaryHDU(header=primary_header),
        fits.ImageHDU(
            data=np.asarray(product.data_map, dtype=float),
            header=header_wcs,
            name="MAP",
        ),
        fits.ImageHDU(
            data=np.asarray(product.count_map, dtype=float),
            header=header_wcs,
            name="COUNTS",
        ),
    ]

    if getattr(product, "std_map", None) is not None:
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(product.std_map, dtype=float),
                header=header_wcs,
                name="STD",
            )
        )

    fits.HDUList(hdus).writeto(file_path, overwrite=True)
    return file_path
