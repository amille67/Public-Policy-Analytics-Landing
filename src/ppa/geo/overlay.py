"""Common spatial overlay operations with stable column naming."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apportion_by_area(
    source: Any,
    target: Any,
    *,
    value_columns: list[str],
    source_id_col: str | None = None,
    target_id_col: str | None = None,
    projected_crs: str = "EPSG:5070",
) -> Any:
    """Areal interpolation from source polygons into target polygons.

    This utility computes intersection areas between non-coterminous polygon
    layers and allocates source counts proportionally by overlap area.

    Args:
        source: Source polygon GeoDataFrame with count/intensity columns.
        target: Target polygon GeoDataFrame to receive apportioned values.
        value_columns: Numeric columns on ``source`` to apportion by area.
        source_id_col: Optional source identifier; a synthetic one is created if absent.
        target_id_col: Optional target identifier; a synthetic one is created if absent.
        projected_crs: Equal-area or projected CRS used for area calculations.

    Returns:
        Target GeoDataFrame with ``value_columns`` replaced by apportioned sums.
    """
    import geopandas as gpd

    if source.empty:
        raise ValueError("source must contain at least one geometry")
    if target.empty:
        raise ValueError("target must contain at least one geometry")

    missing = [c for c in value_columns if c not in source.columns]
    if missing:
        raise ValueError(f"value_columns not found on source: {missing}")

    src = source.copy()
    tgt = target.copy()

    src_id = source_id_col or "_source_id"
    tgt_id = target_id_col or "_target_id"

    if src_id in tgt.columns and src_id != source_id_col:
        raise ValueError(f"column name collision in target for generated source id: {src_id}")
    if tgt_id in src.columns and tgt_id != target_id_col:
        raise ValueError(f"column name collision in source for generated target id: {tgt_id}")

    if source_id_col is None:
        src[src_id] = src.index.astype(str)
    if target_id_col is None:
        tgt[tgt_id] = tgt.index.astype(str)

    # Project into planar CRS for stable area calculations.
    src_proj = src.to_crs(projected_crs)
    tgt_proj = tgt.to_crs(projected_crs)

    src_proj["_source_area"] = src_proj.geometry.area
    if (src_proj["_source_area"] <= 0).any():
        raise ValueError("source polygons must have positive area")

    overlap = gpd.overlay(
        src_proj[[src_id, "_source_area", *value_columns, "geometry"]],
        tgt_proj[[tgt_id, "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if overlap.empty:
        out = tgt.copy()
        for col in value_columns:
            out[col] = 0.0
        return out

    overlap["_intersection_area"] = overlap.geometry.area
    overlap = overlap[overlap["_intersection_area"] > 0].copy()

    overlap["_weight"] = overlap["_intersection_area"] / overlap["_source_area"]
    for col in value_columns:
        overlap[f"_apportioned_{col}"] = overlap[col].astype(float) * overlap["_weight"]

    group_cols = [f"_apportioned_{col}" for col in value_columns]
    apportioned = overlap.groupby(tgt_id, as_index=False)[group_cols].sum()

    out = tgt.merge(apportioned, on=tgt_id, how="left")
    for col in value_columns:
        out[col] = out[f"_apportioned_{col}"].fillna(0.0)
        out = out.drop(columns=[f"_apportioned_{col}"])

    return out


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
