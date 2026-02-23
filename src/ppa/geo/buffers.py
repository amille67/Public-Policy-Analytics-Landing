"""Multiple ring buffer utility (Python equivalent of multipleRingBuffer)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def multiple_ring_buffer(
    input_polygon: Any,
    max_distance: float,
    interval: float,
) -> Any:
    """Create donut-shaped ring buffers around a polygon.

    This is the Python equivalent of the R ``multipleRingBuffer`` helper.
    Produces a GeoDataFrame of ring geometries at successive distance
    intervals from the input polygon.

    Args:
        input_polygon: Shapely Polygon or MultiPolygon geometry (valid required).
        max_distance: Outer boundary distance (same units as geometry CRS).
            May be positive (outward) or negative (inward).
        interval: Step between rings (non-zero; sign must match max_distance
            such that max_distance / interval > 0).

    Returns:
        GeoDataFrame with columns:
            - ``distance`` (float): outer ring distance
            - ``geometry`` (Polygon|MultiPolygon): donut ring geometry
        Sorted ascending by distance if interval > 0, else descending.

    Raises:
        ValueError: If interval is zero, signs are inconsistent, or the
            geometry cannot be buffered.
    """
    import geopandas as gpd
    from shapely.validation import make_valid

    if interval == 0:
        raise ValueError("interval must be non-zero")
    if max_distance / interval < 0:
        raise ValueError(
            f"max_distance ({max_distance}) and interval ({interval}) have inconsistent "
            "signs; progression from 0 must move toward max_distance."
        )

    # Validate / fix geometry
    if not input_polygon.is_valid:
        input_polygon = make_valid(input_polygon)
        if not input_polygon.is_valid:
            raise ValueError("Input polygon is invalid and could not be repaired")

    # Build distance sequence starting at 0
    n_steps = int(abs(max_distance) / abs(interval))
    distances = np.arange(0, (n_steps + 1)) * interval

    rings: list[dict] = []

    for i in range(1, len(distances)):
        d = distances[i]
        prev_d = distances[i - 1]

        if d < 0:
            # Inward (negative) buffer
            buf_d = input_polygon.buffer(d)
            if i == 1:
                ring = input_polygon.difference(buf_d)
            else:
                buf_prev = input_polygon.buffer(prev_d)
                ring = buf_prev.difference(buf_d)
        else:
            # Outward (positive) buffer
            buf_d = input_polygon.buffer(d)
            buf_prev = input_polygon.buffer(prev_d)
            ring = buf_d.difference(buf_prev)

        if ring is None or ring.is_empty:
            continue

        rings.append({"distance": float(d), "geometry": ring})

    if not rings:
        logger.warning("No rings produced; check max_distance and interval parameters")
        return gpd.GeoDataFrame({"distance": [], "geometry": []})

    gdf = gpd.GeoDataFrame(rings)
    gdf = gdf.sort_values("distance", ascending=(interval > 0)).reset_index(drop=True)
    return gdf
