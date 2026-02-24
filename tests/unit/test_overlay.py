"""Unit tests for ppa.geo.overlay."""

from __future__ import annotations

import pytest

from ppa.geo.overlay import apportion_by_area


def test_apportion_by_area_splits_evenly() -> None:
    gpd = pytest.importorskip("geopandas")
    box = pytest.importorskip("shapely.geometry").box

    source = gpd.GeoDataFrame(
        {"units": [100.0]},
        geometry=[box(0, 0, 2, 1)],
        crs="EPSG:4326",
    )
    target = gpd.GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    result = apportion_by_area(source, target, value_columns=["units"])

    assert pytest.approx(result.loc[0, "units"], abs=1e-6) == 50.0
    assert pytest.approx(result.loc[1, "units"], abs=1e-6) == 50.0


def test_apportion_by_area_requires_columns() -> None:
    gpd = pytest.importorskip("geopandas")
    box = pytest.importorskip("shapely.geometry").box

    source = gpd.GeoDataFrame({"x": [1.0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    target = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

    with pytest.raises(ValueError, match="value_columns not found"):
        apportion_by_area(source, target, value_columns=["missing"])
