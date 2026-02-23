"""Land-use and morphology loaders for national tract analytics."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import pandas as pd


def load_building_footprints(path: str | Path) -> gpd.GeoDataFrame:
    """Load Microsoft US Building Footprints from local geospatial file."""
    gdf = gpd.read_file(path)
    return gdf.to_crs(epsg=4326)


def load_padus(path: str | Path) -> gpd.GeoDataFrame:
    """Load PAD-US protected areas from local geospatial file."""
    gdf = gpd.read_file(path)
    return gdf.to_crs(epsg=4326)


def compute_nlcd_impervious_by_tract(
    nlcd_raster_path: str | Path,
    tracts_gdf: gpd.GeoDataFrame,
    *,
    impervious_threshold: int = 20,
) -> pd.DataFrame:
    """Compute tract-level impervious percentage from NLCD raster values."""
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.features import geometry_mask  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio is required for NLCD zonal statistics") from exc

    if "GEOID" not in tracts_gdf.columns:
        raise ValueError("tracts_gdf must include GEOID")

    with rasterio.open(nlcd_raster_path) as src:
        raster = src.read(1)
        valid = raster != src.nodata
        tracts_proj = tracts_gdf.to_crs(src.crs)

        rows: list[dict[str, float | str]] = []
        for row in tracts_proj[["GEOID", "geometry"]].itertuples(index=False):
            try:
                mask = geometry_mask(
                    [row.geometry],
                    transform=src.transform,
                    invert=True,
                    out_shape=raster.shape,
                )
                values = raster[mask & valid]
            except Exception:  # pragma: no cover - geometry/raster edge-case handling
                values = np.array([])

            if values.size == 0:
                pct = np.nan
            else:
                pct = float((values >= impervious_threshold).mean() * 100.0)
            rows.append({"GEOID": row.GEOID, "impervious_pct": pct})

    return pd.DataFrame(rows)
