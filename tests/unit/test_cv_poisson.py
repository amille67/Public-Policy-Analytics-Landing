"""Unit tests for ppa.ml.cv: cross_validate_poisson_by_group."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point


def make_synthetic_gdf(n_per_group: int = 10, n_groups: int = 3, seed: int = 42):
    """Create a synthetic GeoDataFrame with Poisson counts."""
    import geopandas as gpd

    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        x = rng.uniform(0, 100, n_per_group)
        coef = 0.5 * (g + 1)
        lam = np.exp(1.0 + coef * x / 100)
        y = rng.poisson(lam)
        for i in range(n_per_group):
            rows.append({"group_id": g, "y": int(y[i]), "x1": float(x[i]), "geometry": Point(x[i], float(i))})

    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:26918")


class TestCrossValidatePoissonByGroup:
    def test_predictions_filled_all_rows(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=15, n_groups=3)
        result = cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])
        assert "Prediction" in result.columns
        assert result["Prediction"].notna().all()

    def test_preserves_row_order_and_index(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=10, n_groups=2)
        result = cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])
        assert list(result.index) == list(gdf.index)

    def test_preserves_geometry(self) -> None:
        import geopandas as gpd
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=10, n_groups=2)
        result = cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == gdf.crs

    def test_raises_on_negative_counts(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=10, n_groups=2)
        gdf.loc[0, "y"] = -1
        with pytest.raises(ValueError, match="negative"):
            cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])

    def test_raises_on_null_predictor(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=10, n_groups=2)
        gdf.loc[0, "x1"] = np.nan
        with pytest.raises(ValueError, match="null"):
            cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])

    def test_raises_on_single_group(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=10, n_groups=1)
        with pytest.raises(ValueError, match="at least 2"):
            cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])

    def test_predictions_are_positive(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=15, n_groups=3)
        result = cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])
        # Poisson predictions should be non-negative
        assert (result["Prediction"] >= 0).all()

    def test_correlation_with_observed_is_positive(self) -> None:
        from ppa.ml.cv import cross_validate_poisson_by_group

        gdf = make_synthetic_gdf(n_per_group=20, n_groups=3, seed=123)
        result = cross_validate_poisson_by_group(gdf, "group_id", "y", ["x1"])
        corr = float(np.corrcoef(result["y"], result["Prediction"])[0, 1])
        assert corr > 0, f"Expected positive correlation, got {corr:.4f}"
