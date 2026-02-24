"""CRS enforcement and reprojection utilities."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Region -> EPSG code mapping for projected (meter) CRS
REGION_CRS: dict[str, int] = {
    "philadelphia": 26918,
    "lancaster": 26918,
    "boston": 26919,
    "chicago": 26916,
}


def assert_projected_crs(gdf: Any, *, where: str = "") -> None:
    """Assert that a GeoDataFrame has a projected (non-geographic) CRS.

    Args:
        gdf: GeoDataFrame to check.
        where: Descriptive location for error messages.

    Raises:
        ValueError: If CRS is missing or geographic.
    """
    from ppa.util.errors import InvalidCRSError

    if gdf.crs is None:
        raise InvalidCRSError(
            f"GeoDataFrame has no CRS set{' at ' + where if where else ''}"
        )
    if gdf.crs.is_geographic:
        raise InvalidCRSError(
            f"GeoDataFrame has geographic CRS {gdf.crs} at {where!r}. "
            "Reproject to a projected CRS (meters) before distance/buffer operations."
        )


def ensure_crs(gdf: Any, target_epsg: int) -> Any:
    """Ensure a GeoDataFrame is in the target CRS, reprojecting if needed.

    Args:
        gdf: Input GeoDataFrame.
        target_epsg: Target EPSG code (must be projected).

    Returns:
        GeoDataFrame in target CRS.
    """
    if gdf.crs is None:
        logger.warning(
            "GeoDataFrame has no CRS; assuming EPSG:4326 before reprojection"
        )
        gdf = gdf.set_crs(epsg=4326)

    if gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(epsg=target_epsg)
        logger.info("Reprojected to EPSG:%d", target_epsg)
    return gdf


def to_projected_meters(gdf: Any, *, region: str) -> Any:
    """Reproject a GeoDataFrame to the standard meter CRS for a region.

    Args:
        gdf: Input GeoDataFrame.
        region: Region name (one of: philadelphia, lancaster, boston, chicago).

    Returns:
        GeoDataFrame reprojected to the region's standard EPSG.

    Raises:
        ValueError: If region is not in the known mapping.
    """
    key = region.lower()
    if key not in REGION_CRS:
        raise ValueError(f"Unknown region '{region}'. Known: {list(REGION_CRS)}")
    return ensure_crs(gdf, REGION_CRS[key])
