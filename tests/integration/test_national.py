"""Integration tests for national orchestrator (small mocked workload)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd  # type: ignore[import-untyped]
import yaml
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from national.orchestrator import NationalOrchestrator


def _mock_tracts() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "GEOID": ["10001000100", "10001000200", "10001000300", "10001000400", "10001000500"],
            "NAME": [f"Tract {i}" for i in range(1, 6)],
            "total_pop": [100, 120, 140, 160, 180],
            "STATEFP": ["10"] * 5,
        },
        geometry=[
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0)]),
            Polygon([(3, 0), (3, 1), (4, 1), (4, 0)]),
            Polygon([(4, 0), (4, 1), (5, 1), (5, 0)]),
        ],
        crs="EPSG:4326",
    )


def test_national_orchestrator_idempotent(tmp_path: Path) -> None:
    config = {
        "acs_year": 2022,
        "acs_preset": "demographics",
        "peer_msas": [{"name": "Mock", "state_fips": "10", "county_fips": "001"}],
    }
    config_path = tmp_path / "national.yaml"
    config_path.write_text(yaml.safe_dump(config))

    orch = NationalOrchestrator(config_path=config_path, interim_root=tmp_path / "interim")

    with patch("national.orchestrator.fetch_acs_tracts", return_value=_mock_tracts()):
        first = orch.run(dataset="acs_tracts")
        start = time.perf_counter()
        second = orch.run(dataset="acs_tracts")
        elapsed = time.perf_counter() - start

    parquet_files = list((tmp_path / "interim").rglob("*.parquet"))
    assert parquet_files, "Expected parquet outputs"
    assert first["processed"] == 1
    assert second["skipped"] == 1
    assert elapsed < 10
