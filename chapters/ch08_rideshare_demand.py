"""Chapter 8: Predicting Rideshare Demand.

Predicts spatiotemporal rideshare demand in Chicago using time series
features and gradient boosting.

Run:
    python -m chapters.ch08_rideshare_demand --config config/chapters/ch08.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch8 pipeline end-to-end."""
    import numpy as np
    import pandas as pd

    from ppa.io.readers import read_csv
    from ppa.io.writers import write_figure, write_json, write_parquet
    from ppa.ml.metrics import regression_metrics
    from ppa.ml.models import fit_gradient_boosting, save_model
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.themes import plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch08" if output_root else Path("outputs/ch08")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg.inputs
    dt_col = getattr(cfg, "pickup_datetime_col", "trip_start_timestamp")
    spatial_col = getattr(cfg, "spatial_unit_col", "pickup_community_area")
    test_days = int(getattr(cfg, "test_days", 14))

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = read_csv(data_root / inputs["trips"])

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        df = df.head(sample_n)

    # Find datetime column
    for c in [dt_col, "trip_start_timestamp", "pickup_datetime", "started_on"]:
        if c in df.columns:
            dt_col = c
            break

    for c in [spatial_col, "pickup_community_area", "community_area", "zone_id"]:
        if c in df.columns:
            spatial_col = c
            break

    if dt_col not in df.columns:
        raise ValueError(f"Datetime column '{dt_col}' not found. Columns: {list(df.columns)}")

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])

    if spatial_col not in df.columns:
        logger.warning("Spatial column '%s' not found; using 'zone_0'", spatial_col)
        df[spatial_col] = "zone_0"

    df[spatial_col] = df[spatial_col].fillna("unknown").astype(str)
    df["ts_hour"] = df[dt_col].dt.floor("h")

    # ── 2. Aggregate to hourly demand ─────────────────────────────────────────
    demand = (
        df.groupby([spatial_col, "ts_hour"])
        .size()
        .reset_index(name="demand")
    )
    demand = demand.sort_values([spatial_col, "ts_hour"])

    # ── 3. Feature engineering ────────────────────────────────────────────────
    demand["hour"] = demand["ts_hour"].dt.hour
    demand["dow"] = demand["ts_hour"].dt.dayofweek
    demand["weekend"] = (demand["dow"] >= 5).astype(int)

    # Lag features per spatial unit (no future leakage)
    demand = demand.sort_values([spatial_col, "ts_hour"]).copy()
    demand["lag_1h"] = demand.groupby(spatial_col)["demand"].shift(1)
    demand["lag_24h"] = demand.groupby(spatial_col)["demand"].shift(24)
    demand["rollmean_6h"] = (
        demand.groupby(spatial_col)["demand"]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )
    demand["rollmean_24h"] = (
        demand.groupby(spatial_col)["demand"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    )

    # ── 4. Train/test split by time ───────────────────────────────────────────
    max_ts = demand["ts_hour"].max()
    test_cutoff = max_ts - pd.Timedelta(days=test_days)

    train_df = demand[demand["ts_hour"] <= test_cutoff].dropna()
    test_df = demand[demand["ts_hour"] > test_cutoff].dropna()

    feature_cols = ["hour", "dow", "weekend", "lag_1h", "lag_24h", "rollmean_6h", "rollmean_24h"]
    target = "demand"

    if len(train_df) < 5 or len(test_df) == 0:
        logger.warning("Insufficient data after split (train=%d, test=%d)", len(train_df), len(test_df))
        write_parquet(demand, out_dir / "time_series.parquet")
        write_json({"rmse": None, "mae": None}, out_dir / "model_metrics.json")
        return

    model_cfg = getattr(cfg, "model", {}) or {}
    if isinstance(model_cfg, dict):
        n_est = model_cfg.get("n_estimators", 100)
        max_depth = model_cfg.get("max_depth", 3)
    else:
        n_est, max_depth = 100, 3

    model = fit_gradient_boosting(
        train_df[feature_cols].values,
        train_df[target].values,
        n_estimators=n_est,
        max_depth=max_depth,
        seed=settings.seed,
    )
    save_model(model, out_dir / "model.pkl")

    y_pred = model.predict(test_df[feature_cols].values)
    metrics = regression_metrics(test_df[target].values, y_pred)

    preds_df = pd.DataFrame({
        spatial_col: test_df[spatial_col].values,
        "ts": test_df["ts_hour"].values,
        "y_true": test_df[target].values,
        "y_pred": y_pred,
    })

    # ── 5. Save ───────────────────────────────────────────────────────────────
    write_parquet(demand, out_dir / "time_series.parquet")
    write_parquet(preds_df, out_dir / "predictions.parquet")
    write_json(metrics, out_dir / "model_metrics.json")

    # Figure: top 3 spatial units
    try:
        import matplotlib.pyplot as plt
        ptheme = plot_theme(title_size=14)
        top_units = preds_df.groupby(spatial_col)["y_true"].sum().nlargest(3).index.tolist()
        with plt.rc_context(ptheme):
            fig, axes = plt.subplots(len(top_units), 1, figsize=(12, 4 * len(top_units)))
            if len(top_units) == 1:
                axes = [axes]
            for ax, unit in zip(axes, top_units):
                sub = preds_df[preds_df[spatial_col] == unit].sort_values("ts")
                ax.plot(sub["ts"], sub["y_true"], label="Actual")
                ax.plot(sub["ts"], sub["y_pred"], label="Predicted", linestyle="--")
                ax.set_title(f"Unit: {unit}")
                ax.legend()
            plt.tight_layout()
        write_figure(fig, fig_dir / "demand_forecast.png")
    except Exception as e:
        logger.warning("Figure error: %s", e)

    logger.info("Ch08 complete. RMSE=%.2f MAE=%.2f", metrics["rmse"], metrics["mae"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch08: Rideshare Demand")
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
        logger.exception("Ch08 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
