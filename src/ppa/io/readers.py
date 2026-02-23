"""Dataset readers with schema validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def read_geodataframe(path: Path, **kwargs: Any) -> Any:
    """Read a geospatial file into a GeoDataFrame.

    Args:
        path: Path to the file (GeoJSON, shapefile dir, geoparquet, etc.).
        **kwargs: Additional kwargs passed to geopandas.read_file.

    Returns:
        GeoDataFrame with at least one geometry column.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has no valid geometries.
    """
    import geopandas as gpd

    if not Path(path).exists():
        raise FileNotFoundError(f"Geospatial file not found: {path}")

    gdf = gpd.read_file(str(path), **kwargs)
    if gdf.empty:
        raise ValueError(f"Empty GeoDataFrame read from {path}")
    logger.info("Read %d rows from %s (CRS: %s)", len(gdf), path, gdf.crs)
    return gdf


def read_csv(
    path: Path,
    *,
    dtypes: dict[str, str] | None = None,
    parse_dates: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.
        dtypes: Optional column dtype overrides.
        parse_dates: Optional list of columns to parse as dates.
        **kwargs: Additional kwargs for pd.read_csv.

    Returns:
        DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df: pd.DataFrame = pd.read_csv(  # type: ignore[assignment]
        path,
        dtype=dtypes,  # type: ignore[arg-type]
        parse_dates=parse_dates or [],
        **kwargs,
    )
    logger.info("Read %d rows, %d cols from %s", len(df), len(df.columns), path)
    return df


def validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Validate that all required columns are present in a DataFrame.

    Args:
        df: DataFrame to check.
        required: List of required column names.
        source: Descriptive name for the data source (for error messages).

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {source}. "
            f"Available: {list(df.columns)}"
        )
