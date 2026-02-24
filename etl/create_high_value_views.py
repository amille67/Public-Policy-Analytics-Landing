"""Materialize exactly 7 pre-computed Nashville high-value views.

Usage
-----
python -m etl.create_high_value_views --views all
python -m etl.create_high_value_views --views 1,3,7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from national.loaders.tiger import tiger_tracts
from ppa.geo.overlay import apportion_by_area
from ppa.io.writers import write_geoparquet, write_parquet


RUNNER_MAP = {
    "01": "run_view_01_nashville_311_risk",
    "02": "run_view_02_nashville_permits_transition",
    "03": "run_view_03_nashville_property_value_equity",
    "04": "run_view_04_nashville_urban_services_district",
    "05": "run_view_05_nashville_industrial_footprints",
    "06": "run_view_06_nashville_transit_access_equity",
    "07": "run_view_07_national_housing_context",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_config(path: Path) -> list[dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    views = config.get("views", [])
    if len(views) != 7:
        raise ValueError("views_config.json must define exactly 7 views for MVP")
    return views


def _resolve_runner(runner_id: str) -> Any:
    from etl.runners import view_runners
    val = RUNNER_MAP[runner_id]
    if callable(val):
        return val
    return getattr(view_runners, val)


def _rollup_county(lowest: Any, view_id: int, fips_list: list[str] | None = None) -> Any:
    if fips_list is None:
        fips_list = ["47037"]

    if "geometry" not in lowest.columns or lowest.geometry.is_empty.all():
        # non-spatial View 7
        county = pd.DataFrame(lowest).copy()
        if "GEOID" in county.columns:
            county["county_fips"] = county["GEOID"].astype(str).str[:5]
        if "county_fips" in county.columns:
            numeric = [c for c in county.columns if pd.api.types.is_numeric_dtype(county[c])]
            county = county.groupby("county_fips", as_index=False)[numeric].sum()
    else:
        if "GEOID" in lowest.columns:
            work = lowest.copy()
            work["county_fips"] = work["GEOID"].astype(str).str[:5]
            numeric_cols = [
                c for c in work.columns
                if pd.api.types.is_numeric_dtype(work[c]) and c not in ("county_fips", "GEOID", "geometry")
            ]
            agg_dict = {c: "sum" for c in numeric_cols}
            county = work.dissolve(by="county_fips", aggfunc=agg_dict if agg_dict else "first", as_index=False)
        else:
            county_polys = tiger_tracts(fips_list)[["GEOID", "geometry"]].copy()
            county_polys["county_fips"] = county_polys["GEOID"].astype(str).str[:5]
            county_polys = county_polys.dissolve(by="county_fips", as_index=False)
            numeric_cols = [c for c in lowest.columns if pd.api.types.is_numeric_dtype(lowest[c])]
            county = apportion_by_area(
                lowest, county_polys[["county_fips", "geometry"]],
                value_columns=numeric_cols, target_id_col="county_fips"
            )

    county["view_id"] = view_id
    county["geo_level"] = "county"
    county["tags"] = json.dumps({"city": "Nashville", "view_id": int(view_id)})
    return county


def _annotate_lowest(df: Any, view_id: int) -> Any:
    df = df.copy()
    df["view_id"] = view_id
    df["geo_level"] = "lowest"
    df["tags"] = json.dumps({"city": "Nashville", "view_id": int(view_id)})
    return df


def _save_outputs(lowest: Any, county: Any, view_id: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    low_path = out_dir / f"view_{view_id:02d}_lowest.geoparquet"
    county_path = out_dir / f"view_{view_id:02d}_county.geoparquet"

    if hasattr(lowest, "geometry") and "geometry" in lowest.columns and not lowest.geometry.is_empty.all():
        write_geoparquet(lowest, low_path)
    else:
        write_parquet(pd.DataFrame(lowest), low_path)

    if hasattr(county, "geometry") and "geometry" in county.columns and not county.geometry.is_empty.all():
        write_geoparquet(county, county_path)
    else:
        write_parquet(pd.DataFrame(county), county_path)


def run_selected(
    views: str,
    *,
    config_path: Path = Path("etl/views_config.json"),
    out_dir: Path | None = None,
) -> list[int]:
    if out_dir is None:
        out_dir = _project_root() / "data" / "views"

    requested = None if views == "all" else {int(x.strip()) for x in views.split(",") if x.strip()}
    executed: list[int] = []

    for spec in _load_config(config_path):
        view_id = int(spec["view_id"])
        if requested is not None and view_id not in requested:
            continue

        runner = _resolve_runner(spec["runner_id"])
        lowest = _annotate_lowest(runner(), view_id)
        county = _rollup_county(lowest, view_id, fips_list=["47037"])

        if len(county) == 0:
            raise ValueError(f"View {view_id}: county roll-up produced zero rows")

        # only validate geometry on spatial views
        if hasattr(lowest, "geometry") and "geometry" in lowest.columns:
            if not bool(lowest.geometry.is_valid.all()):
                raise ValueError(f"View {view_id}: invalid geometries detected")

        _save_outputs(lowest, county, view_id, out_dir)
        executed.append(view_id)

    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nashville high-value pre-computed views")
    parser.add_argument("--views", default="all", help="all or comma-separated IDs (e.g., 1,3,7)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    run_selected(args.views, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
