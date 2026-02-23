"""Common spatial overlay operations with stable column naming."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def clip(gdf: Any, mask: Any) -> Any:
    """Clip a GeoDataFrame to the bounds of a mask geometry/GeoDataFrame.

    Args:
        gdf: GeoDataFrame to clip.
        mask: GeoDataFrame or geometry used as clip boundary.

    Returns:
        Clipped GeoDataFrame (same CRS as input).
    """
    import geopandas as gpd

    result = gpd.clip(gdf, mask)
    logger.info("Clipped from %d to %d rows", len(gdf), len(result))
    return result


def sjoin(
    left: Any,
    right: Any,
    how: str = "left",
    predicate: str = "intersects",
) -> Any:
    """Spatial join wrapper with stable column naming.

    Args:
        left: Left GeoDataFrame.
        right: Right GeoDataFrame.
        how: Join type ('left', 'right', 'inner').
        predicate: Spatial predicate ('intersects', 'within', 'contains').

    Returns:
        Joined GeoDataFrame.
    """
    import geopandas as gpd

    result = gpd.sjoin(left, right, how=how, predicate=predicate)
    # Drop the join index column that geopandas adds
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])
    if "index_left" in result.columns:
        result = result.drop(columns=["index_left"])
    return result


def sjoin_nearest(
    left: Any,
    right: Any,
    how: str = "left",
    max_distance: float | None = None,
    distance_col: str | None = None,
) -> Any:
    """Nearest spatial join with fallback.

    Args:
        left: Left GeoDataFrame.
        right: Right GeoDataFrame.
        how: Join type.
        max_distance: Maximum search distance (units of CRS).
        distance_col: If provided, add a column with the join distance.

    Returns:
        Joined GeoDataFrame.
    """
    import geopandas as gpd

    kwargs: dict[str, Any] = {"how": how}
    if max_distance is not None:
        kwargs["max_distance"] = max_distance
    if distance_col is not None:
        kwargs["distance_col"] = distance_col

    result = gpd.sjoin_nearest(left, right, **kwargs)
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])
    return result
