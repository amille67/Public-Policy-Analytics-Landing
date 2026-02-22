"""Raster-to-DataFrame conversion (Python equivalent of R's rast function)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rast_to_df(dataset: Any, *, band: int = 1) -> pd.DataFrame:
    """Convert a rasterio dataset to a long-format DataFrame of (x, y, value).

    Python equivalent of R ``rast(inRaster)`` which calls ``xyFromCell`` and
    ``getValues`` to produce a flat dataframe suitable for ggplot.

    Args:
        dataset: Open rasterio DatasetReader (single-band recommended).
        band: 1-based band index to read. Defaults to 1.

    Returns:
        DataFrame with columns:
            - ``x`` (float): cell center x coordinate (in CRS units)
            - ``y`` (float): cell center y coordinate (in CRS units)
            - ``value`` (float): pixel value; nodata is converted to NaN

    Raises:
        ValueError: If the requested band index is out of range.
    """
    if band < 1 or band > dataset.count:
        raise ValueError(
            f"Band {band} out of range; dataset has {dataset.count} band(s)"
        )

    arr = dataset.read(band).astype(float)
    transform = dataset.transform
    nodata = dataset.nodata

    rows, cols = arr.shape
    row_idx, col_idx = np.mgrid[0:rows, 0:cols]

    # Affine: cell center = transform * (col + 0.5, row + 0.5)
    x = transform.c + (col_idx + 0.5) * transform.a + (row_idx + 0.5) * transform.b
    y = transform.f + (col_idx + 0.5) * transform.d + (row_idx + 0.5) * transform.e

    values = arr.ravel()
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)

    return pd.DataFrame(
        {
            "x": x.ravel(),
            "y": y.ravel(),
            "value": values,
        }
    )
