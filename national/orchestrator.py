"""National ETL orchestrator with idempotent parquet checkpoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd  # type: ignore[import-untyped]
import yaml
from rich.progress import track

from ppa.data import fetch_acs_tracts
from ppa.io.lakehouse import (
    checkpoint_exists,
    validate_has_parquet,
    validate_no_geojson,
    write_state_partition,
)

if TYPE_CHECKING:  # pragma: no cover
    import dask_geopandas as dgpd


@dataclass(frozen=True)
class StateSpec:
    """State workload definition."""

    state_fips: str
    county_fips: str | None = None


class NationalOrchestrator:
    """Run national/state ETL with idempotent state checkpoints."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        interim_root: str | Path = "data/interim",
    ) -> None:
        self.config_path = Path(config_path)
        self.interim_root = Path(interim_root)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _state_specs(self) -> list[StateSpec]:
        peers = self.config.get("peer_msas", [])
        return [
            StateSpec(
                state_fips=str(peer["state_fips"]).zfill(2),
                county_fips=(
                    str(peer["county_fips"]).zfill(3)
                    if peer.get("county_fips")
                    else None
                ),
            )
            for peer in peers
        ]

    def run(
        self, *, dataset: str = "acs_tracts", force: bool = False
    ) -> dict[str, float | int]:
        """Run state-level ETL ingestion and write parquet partitions."""
        start = time.perf_counter()
        specs = self._state_specs()
        year = int(self.config.get("acs_year", 2022))
        preset = self.config.get("acs_preset", "demographics")

        processed = 0
        skipped = 0

        for spec in track(specs, description="Processing states..."):
            if (
                checkpoint_exists(spec.state_fips, dataset, self.interim_root)
                and not force
            ):
                skipped += 1
                continue

            gdf = fetch_acs_tracts(
                year,
                spec.state_fips,
                county_fips=spec.county_fips,
                preset=preset,
                api_key=os.environ.get("CENSUS_API_KEY"),
            )
            if not gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                raise ValueError("Geometry check failed: expected Polygon/MultiPolygon")

            write_state_partition(
                gdf,
                state_fips=spec.state_fips,
                dataset=dataset,
                root=self.interim_root,
            )
            processed += 1

        validate_no_geojson(self.interim_root)
        validate_has_parquet(self.interim_root)
        runtime_s = time.perf_counter() - start
        return {
            "processed": processed,
            "skipped": skipped,
            "runtime_s": runtime_s,
        }

    def load_analytical_view(
        self,
        dataset: str = "acs_tracts",
        *,
        state_fips: str | None = None,
        county_fips_list: list[str] | None = None,
    ) -> dgpd.GeoDataFrame:
        """Load cross-state analytical view lazily via dask-geopandas.

        Notes
        -----
        Tract parquet partitions do not natively include ``CBSAFP``. Callers
        must resolve CBSA identifiers to county FIPS and pass those as
        ``county_fips_list``.
        """
        import dask_geopandas as dgpd  # type: ignore[import-untyped]

        path_glob = str(self.interim_root / dataset / "STATEFP=*" / "*.parquet")

        # PyArrow semantics: outer list=OR, inner list=AND.
        filters: list[list[tuple[str, str, str]]] = []
        if county_fips_list:
            if state_fips:
                filters = [
                    [
                        ("STATEFP", "==", state_fips.zfill(2)),
                        ("COUNTYFP", "==", county_fips.zfill(3)),
                    ]
                    for county_fips in county_fips_list
                ]
            else:
                filters = [
                    [("COUNTYFP", "==", county_fips.zfill(3))]
                    for county_fips in county_fips_list
                ]
        elif state_fips:
            filters = [[("STATEFP", "==", state_fips.zfill(2))]]

        return dgpd.read_parquet(path_glob, filters=filters if filters else None)

    def load_analytical_geoview(
        self,
        dataset: str = "acs_tracts",
        *,
        state_fips: str | None = None,
        county_fips_list: list[str] | None = None,
    ) -> gpd.GeoDataFrame:
        """Load a filtered analytical geoview to memory.

        At least one spatial filter is required to prevent whole-nation OOM on
        CI runners.
        """
        if state_fips is None and not county_fips_list:
            raise ValueError(
                "load_analytical_geoview requires state_fips or county_fips_list filter"
            )

        ddf = self.load_analytical_view(
            dataset,
            state_fips=state_fips,
            county_fips_list=county_fips_list,
        )
        return ddf.compute()
