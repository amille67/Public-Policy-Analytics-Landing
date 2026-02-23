"""Unit tests for ppa.geo.buffers: multiple_ring_buffer."""

import pytest

from ppa.geo.buffers import multiple_ring_buffer


def make_square(size: float = 100.0):
    """Create a simple square polygon centered at origin."""
    from shapely.geometry import box

    return box(0, 0, size, size)


class TestMultipleRingBuffer:
    def test_positive_rings_non_overlapping(self) -> None:
        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=300, interval=100)
        assert len(gdf) == 3
        # Rings should not overlap: union area ≈ buffer(300) - buffer(0) area
        total_ring_area = gdf.geometry.area.sum()
        expected_area = poly.buffer(300).area - poly.buffer(0).area
        assert abs(total_ring_area - expected_area) / expected_area < 0.01

    def test_distance_column_exact(self) -> None:
        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=300, interval=100)
        assert "distance" in gdf.columns
        distances = sorted(gdf["distance"].tolist())
        assert distances == [100.0, 200.0, 300.0]

    def test_sorted_ascending_for_positive_interval(self) -> None:
        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=400, interval=100)
        assert gdf["distance"].is_monotonic_increasing

    def test_interval_zero_raises(self) -> None:
        poly = make_square(100.0)
        with pytest.raises(ValueError, match="non-zero"):
            multiple_ring_buffer(poly, max_distance=100, interval=0)

    def test_inconsistent_signs_raises(self) -> None:
        poly = make_square(100.0)
        with pytest.raises(ValueError, match="inconsistent"):
            multiple_ring_buffer(poly, max_distance=1000, interval=-250)

    def test_returns_geodataframe(self) -> None:
        import geopandas as gpd

        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=200, interval=100)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_ring_count_matches_steps(self) -> None:
        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=500, interval=100)
        assert len(gdf) == 5

    def test_geometries_are_not_empty(self) -> None:
        poly = make_square(100.0)
        gdf = multiple_ring_buffer(poly, max_distance=300, interval=100)
        assert gdf.geometry.is_empty.sum() == 0

    def test_negative_interval_inward_rings(self) -> None:
        # Inward (negative) buffers on a large polygon
        from shapely.geometry import box

        big_poly = box(0, 0, 1000, 1000)
        gdf = multiple_ring_buffer(big_poly, max_distance=-200, interval=-100)
        assert len(gdf) == 2
        # Inward rings should have area less than outer polygon
        assert gdf.geometry.area.sum() < big_poly.area

    def test_interval_2_steps(self) -> None:
        poly = make_square(10.0)
        gdf = multiple_ring_buffer(poly, max_distance=20, interval=10)
        assert len(gdf) == 2
