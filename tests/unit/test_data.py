"""Tests for ppa.data — ACS fetcher and CRS standardizer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd
import pytest
from shapely.geometry import Point  # type: ignore[import-untyped]

from ppa.data import (  # isort: skip
    ACS_VARIABLE_PRESETS,
    _DEFAULT_STORAGE_EPSG,
    fetch_acs_tracts,
    load_geojson,
    standardize_crs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_gdf_4326() -> gpd.GeoDataFrame:
    """GeoDataFrame in EPSG:4326 with three points."""
    return gpd.GeoDataFrame(
        {"value": [10, 20, 30]},
        geometry=[Point(-75.16, 39.95), Point(-75.17, 39.96), Point(-75.18, 39.97)],
        crs="EPSG:4326",
    )


@pytest.fixture()
def sample_gdf_26918() -> gpd.GeoDataFrame:
    """GeoDataFrame in EPSG:26918 (UTM 18N, meters)."""
    return gpd.GeoDataFrame(
        {"value": [1, 2]},
        geometry=[Point(500000, 4400000), Point(500100, 4400100)],
        crs="EPSG:26918",
    )


@pytest.fixture()
def sample_gdf_no_crs() -> gpd.GeoDataFrame:
    """GeoDataFrame with no CRS set."""
    return gpd.GeoDataFrame(
        {"value": [1]},
        geometry=[Point(0, 0)],
    )


MOCK_ACS_RESPONSE: list[list[str]] = [
    ["NAME", "B01003_001E", "B02001_002E", "B03001_003E", "B01002_001E", "state", "county", "tract"],
    ["Tract 1, County, State", "5000", "3000", "1000", "35.5", "42", "101", "000100"],
    ["Tract 2, County, State", "8000", "4500", "2000", "42.1", "42", "101", "000200"],
    ["Tract 3, County, State", "3000", "2000", "500", "28.0", "42", "101", "000300"],
]


# ---------------------------------------------------------------------------
# Tests: standardize_crs
# ---------------------------------------------------------------------------


class TestStandardizeCrs:
    """Tests for the standardize_crs function."""

    def test_already_target_crs(self, sample_gdf_4326: gpd.GeoDataFrame) -> None:
        """No reprojection when already in target CRS."""
        result = standardize_crs(sample_gdf_4326, target_epsg=4326)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326
        assert len(result) == 3

    def test_reproject_to_4326(self, sample_gdf_26918: gpd.GeoDataFrame) -> None:
        """Reproject from UTM 18N to WGS 84."""
        result = standardize_crs(sample_gdf_26918, target_epsg=4326)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326
        # Coordinates should now be in lon/lat range
        xs = result.geometry.x.tolist()
        assert all(-180 <= x <= 180 for x in xs)

    def test_reproject_to_projected(self, sample_gdf_4326: gpd.GeoDataFrame) -> None:
        """Reproject from WGS 84 to a projected CRS."""
        result = standardize_crs(sample_gdf_4326, target_epsg=26918)
        assert result.crs is not None
        assert result.crs.to_epsg() == 26918

    def test_no_crs_with_source(self, sample_gdf_no_crs: gpd.GeoDataFrame) -> None:
        """Accept source_epsg when the GeoDataFrame has no CRS."""
        result = standardize_crs(
            sample_gdf_no_crs,
            target_epsg=4326,
            source_epsg=4326,
        )
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_no_crs_no_source_raises(self, sample_gdf_no_crs: gpd.GeoDataFrame) -> None:
        """Raise ValueError when CRS is missing and no source_epsg given."""
        with pytest.raises(ValueError, match="no CRS"):
            standardize_crs(sample_gdf_no_crs, target_epsg=4326)

    def test_preserves_data(self, sample_gdf_26918: gpd.GeoDataFrame) -> None:
        """Non-geometry columns survive reprojection."""
        result = standardize_crs(sample_gdf_26918, target_epsg=4326)
        assert list(result["value"]) == [1, 2]


# ---------------------------------------------------------------------------
# Tests: fetch_acs_tracts
# ---------------------------------------------------------------------------


class TestFetchAcsTracts:
    """Tests for the fetch_acs_tracts function, with mocked API calls."""

    @patch("ppa.data._fetch_acs_raw")
    def test_basic_fetch(self, mock_raw: MagicMock) -> None:
        """Fetch tracts with the default 'demographics' preset."""
        mock_raw.return_value = MOCK_ACS_RESPONSE

        gdf = fetch_acs_tracts(2022, "42", county_fips="101")

        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert "GEOID" in gdf.columns
        assert "total_pop" in gdf.columns
        assert "pop_white" in gdf.columns
        # GEOID should be state+county+tract
        assert gdf.iloc[0]["GEOID"] == "42101000100"

    @patch("ppa.data._fetch_acs_raw")
    def test_numeric_conversion(self, mock_raw: MagicMock) -> None:
        """ACS values should be converted to numeric types."""
        mock_raw.return_value = MOCK_ACS_RESPONSE

        gdf = fetch_acs_tracts(2022, "42", county_fips="101")

        assert pd.api.types.is_numeric_dtype(gdf["total_pop"])
        assert gdf.iloc[0]["total_pop"] == 5000

    @patch("ppa.data._fetch_acs_raw")
    def test_custom_variables(self, mock_raw: MagicMock) -> None:
        """Fetch with explicit variable mapping."""
        custom_response: list[list[str]] = [
            ["NAME", "B19013_001E", "state", "county", "tract"],
            ["Tract 1", "55000", "42", "101", "000100"],
        ]
        mock_raw.return_value = custom_response

        gdf = fetch_acs_tracts(
            2022,
            "42",
            county_fips="101",
            variables={"B19013_001E": "median_income"},
        )

        assert "median_income" in gdf.columns
        assert gdf.iloc[0]["median_income"] == 55000

    @patch("ppa.data._fetch_acs_raw")
    def test_crs_is_4326(self, mock_raw: MagicMock) -> None:
        """Returned GeoDataFrame should be in EPSG:4326."""
        mock_raw.return_value = MOCK_ACS_RESPONSE

        gdf = fetch_acs_tracts(2022, "42", county_fips="101")

        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == _DEFAULT_STORAGE_EPSG

    def test_unknown_preset_raises(self) -> None:
        """Raise ValueError for an unknown preset name."""
        with pytest.raises(ValueError, match="Unknown ACS preset"):
            fetch_acs_tracts(2022, "42", preset="nonexistent")

    def test_no_variables_no_preset_raises(self) -> None:
        """Raise ValueError when neither variables nor preset is given."""
        with pytest.raises(ValueError, match="Either"):
            fetch_acs_tracts(2022, "42", variables=None, preset=None)

    @patch("ppa.data._fetch_acs_raw")
    def test_api_key_from_env(self, mock_raw: MagicMock) -> None:
        """API key falls back to CENSUS_API_KEY env var."""
        mock_raw.return_value = MOCK_ACS_RESPONSE

        with patch.dict("os.environ", {"CENSUS_API_KEY": "test-key-123"}):
            fetch_acs_tracts(2022, "42", county_fips="101")

        # Verify the cached function was called with the env-resolved key
        mock_raw.assert_called_once()
        call_args = mock_raw.call_args
        # _fetch_acs_raw is called with keyword args
        if call_args.kwargs:
            assert call_args.kwargs["api_key"] == "test-key-123"
        else:
            # Positional: (year, variables, state_fips, county_fips, api_key)
            assert call_args.args[4] == "test-key-123"


# ---------------------------------------------------------------------------
# Tests: load_geojson
# ---------------------------------------------------------------------------


class TestLoadGeojson:
    """Tests for the load_geojson function."""

    def test_file_not_found(self, tmp_path: Any) -> None:
        """Raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_geojson(tmp_path / "nonexistent.geojson")

    def test_loads_and_standardizes(self, tmp_path: Any) -> None:
        """Load a GeoJSON file and verify CRS standardization."""
        # Write a small GeoJSON in EPSG:4326
        gdf_src = gpd.GeoDataFrame(
            {"name": ["A", "B"]},
            geometry=[Point(-75.16, 39.95), Point(-75.17, 39.96)],
            crs="EPSG:4326",
        )
        out_path = tmp_path / "test.geojson"
        gdf_src.to_file(str(out_path), driver="GeoJSON")

        result = load_geojson(out_path)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_loads_with_reprojection(self, tmp_path: Any) -> None:
        """Load a GeoJSON file and reproject to a different CRS."""
        gdf_src = gpd.GeoDataFrame(
            {"val": [1]},
            geometry=[Point(-75.16, 39.95)],
            crs="EPSG:4326",
        )
        out_path = tmp_path / "proj.geojson"
        gdf_src.to_file(str(out_path), driver="GeoJSON")

        result = load_geojson(out_path, target_epsg=26918)

        assert result.crs is not None
        assert result.crs.to_epsg() == 26918


# ---------------------------------------------------------------------------
# Tests: ACS presets
# ---------------------------------------------------------------------------


class TestAcsPresets:
    """Validate structure of built-in ACS variable presets."""

    def test_presets_are_non_empty(self) -> None:
        """Each preset should have at least one variable."""
        for name, mapping in ACS_VARIABLE_PRESETS.items():
            assert len(mapping) > 0, f"Preset {name!r} is empty"

    def test_preset_keys_look_like_census_codes(self) -> None:
        """Census variable codes should match B\\d+_\\d+E pattern."""
        import re

        pattern = re.compile(r"^B\d{5}_\d{3}E$")
        for name, mapping in ACS_VARIABLE_PRESETS.items():
            for code in mapping:
                assert pattern.match(code), (
                    f"Preset {name!r}: {code!r} does not look like a Census variable code"
                )

    def test_preset_values_are_snake_case(self) -> None:
        """Column names should be snake_case identifiers."""
        import re

        pattern = re.compile(r"^[a-z][a-z0-9_]*$")
        for name, mapping in ACS_VARIABLE_PRESETS.items():
            for col_name in mapping.values():
                assert pattern.match(col_name), (
                    f"Preset {name!r}: column {col_name!r} is not snake_case"
                )
