"""Chapter 2: Expanding the Urban Growth Boundary.

Quantifies development and greenspace patterns relative to Lancaster
County's Urban Growth Boundary using ring buffers and spatial overlays.

Run:
    python -m chapters.ch02_ugb_sprawl --config config/chapters/ch02.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch2 pipeline end-to-end."""
    import pandas as pd

    from ppa.geo.buffers import multiple_ring_buffer
    from ppa.geo.crs import ensure_crs
    from ppa.geo.overlay import clip
    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_geodataframe
    from ppa.io.writers import write_csv, write_figure, write_geoparquet, write_parquet
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.maps import choropleth_map
    from ppa.viz.themes import map_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch02" if output_root else chapter_output_dir("ch02")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epsg = getattr(cfg, "crs_epsg", 26918) or 26918
    inputs = cfg.inputs

    # ── 1. Load ───────────────────────────────────────────────────────────────
    logger.info("Loading Ch02 layers...")
    towns = read_geodataframe(data_root / inputs["towns"])
    ugb = read_geodataframe(data_root / inputs["ugb"])
    buildings = read_geodataframe(data_root / inputs["buildings"])
    boundary = read_geodataframe(data_root / inputs["boundary"])
    greenspace = read_geodataframe(data_root / inputs["greenspace"])

    # ── 2. Reproject ──────────────────────────────────────────────────────────
    for name, _gdf in [
        ("towns", towns),
        ("ugb", ugb),
        ("buildings", buildings),
        ("boundary", boundary),
        ("greenspace", greenspace),
    ]:
        logger.info("Reprojecting %s...", name)

    towns = ensure_crs(towns, epsg)
    ugb = ensure_crs(ugb, epsg)
    buildings = ensure_crs(buildings, epsg)
    boundary = ensure_crs(boundary, epsg)
    greenspace = ensure_crs(greenspace, epsg)

    # Clip to county boundary
    buildings_clipped = clip(buildings, boundary)
    greenspace_clipped = clip(greenspace, boundary)

    # ── 3. Ring buffers ───────────────────────────────────────────────────────
    max_dist = float(getattr(cfg, "max_distance_m", 5000))
    interval = float(getattr(cfg, "interval_m", 500))
    ugb_geom = ugb.unary_union

    logger.info("Building ring buffers: max=%dm interval=%dm", max_dist, interval)
    rings_gdf = multiple_ring_buffer(ugb_geom, max_dist, interval)
    rings_gdf = rings_gdf.set_crs(epsg=epsg)

    # ── 4. Summarize buildings and greenspace by ring ─────────────────────────
    ring_rows = []
    for _, ring_row in rings_gdf.iterrows():
        d = ring_row["distance"]
        ring_geom = ring_row["geometry"]
        b_in_ring = buildings_clipped[buildings_clipped.geometry.intersects(ring_geom)]
        g_in_ring = greenspace_clipped[
            greenspace_clipped.geometry.intersects(ring_geom)
        ]

        b_count = len(b_in_ring)
        b_area = (
            float(b_in_ring.geometry.area.sum())
            if len(b_in_ring) > 0
            and b_in_ring.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any()
            else None
        )
        g_area = float(g_in_ring.geometry.area.sum()) if len(g_in_ring) > 0 else 0.0

        ring_rows.append(
            {
                "distance": d,
                "building_count": b_count,
                "building_area_m2": b_area,
                "greenspace_area_m2": g_area,
            }
        )

    ring_metrics = pd.DataFrame(ring_rows)

    # ── 5. Town metrics ───────────────────────────────────────────────────────
    town_id_col = getattr(cfg, "town_id_col", None)
    if town_id_col not in towns.columns:
        for cand in ["NAME", "name", "NAMELSAD", "town_id"]:
            if cand in towns.columns:
                town_id_col = cand
                break

    # Compute inside/outside UGB building counts per town
    town_metrics_rows = []
    for _, town in towns.iterrows():
        town_geom = town.geometry
        b_in_town = buildings_clipped[buildings_clipped.geometry.intersects(town_geom)]
        b_inside = buildings_clipped[
            buildings_clipped.geometry.intersects(town_geom)
            & buildings_clipped.geometry.intersects(ugb_geom)
        ]
        inside_cnt = len(b_inside)
        outside_cnt = len(b_in_town) - inside_cnt
        sprawl = outside_cnt / inside_cnt if inside_cnt > 0 else float("nan")

        town_metrics_rows.append(
            {
                "town_id": str(town[town_id_col]) if town_id_col else str(town.name),
                "buildings_inside_ugb": inside_cnt,
                "buildings_outside_ugb": outside_cnt,
                "sprawl_index": sprawl,
            }
        )

    town_metrics = pd.DataFrame(town_metrics_rows)

    # ── 6. Outputs ────────────────────────────────────────────────────────────
    write_geoparquet(rings_gdf, out_dir / "rings.geoparquet")
    write_parquet(ring_metrics, out_dir / "ring_metrics.parquet")
    write_csv(town_metrics, out_dir / "town_metrics.csv")

    # Figures
    mtheme = map_theme(title_size=24)
    try:
        if len(rings_gdf) > 0 and "distance" in rings_gdf.columns:
            fig = choropleth_map(
                rings_gdf, "distance", title="UGB Ring Buffers", theme=mtheme
            )
            write_figure(fig, fig_dir / "ugb_rings.png")
    except Exception as e:
        logger.warning("Ring map error: %s", e)

    try:
        if not town_metrics.empty and "sprawl_index" in town_metrics.columns:
            towns_merged = towns.merge(
                town_metrics,
                left_on=town_id_col or "NAME",
                right_on="town_id",
                how="left",
            )
            fig2 = choropleth_map(
                towns_merged, "sprawl_index", title="Town Sprawl Index", theme=mtheme
            )
            write_figure(fig2, fig_dir / "town_sprawl_index.png")
    except Exception as e:
        logger.warning("Town sprawl map error: %s", e)

    logger.info("Ch02 pipeline complete. Outputs in %s", out_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ch02: Urban Growth Boundary Analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    from ppa.util.config import load_chapter_config, load_settings
    from ppa.util.logging import get_logger

    settings = load_settings()
    cfg = load_chapter_config(Path(args.config), settings)
    if args.sample:
        cfg.sample = args.sample

    get_logger(__name__, settings.log_level)
    output_root = Path(args.output_root) if args.output_root else None

    try:
        build_pipeline(cfg, settings, output_root=output_root)
        return 0
    except Exception:
        logger.exception("Ch02 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
