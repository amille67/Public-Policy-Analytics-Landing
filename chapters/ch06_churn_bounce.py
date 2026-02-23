"""Chapter 6: People-Based ML Models (Churn Prediction).

Predicts churn and performs cost/benefit threshold analysis using
logistic regression and iterateThresholds sweep.

Run:
    python -m chapters.ch06_churn_bounce --config config/chapters/ch06.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_pipeline(cfg: Any, settings: Any, output_root: Path | None = None) -> None:
    """Execute the Ch6 pipeline end-to-end."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    from ppa.io.paths import chapter_output_dir
    from ppa.io.readers import read_csv
    from ppa.io.writers import write_csv, write_figure, write_json, write_parquet
    from ppa.ml.metrics import classification_metrics
    from ppa.ml.models import fit_logistic_regression, save_model
    from ppa.ml.thresholds import iterate_thresholds
    from ppa.util.reproducibility import set_global_seed
    from ppa.viz.plots import utility_by_threshold
    from ppa.viz.themes import plot_theme

    set_global_seed(settings.seed)

    data_root = Path(settings.data_root)
    out_dir = output_root / "ch06" if output_root else chapter_output_dir("ch06")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs = cfg.inputs

    # ── 1. Load ───────────────────────────────────────────────────────────────
    churn_df = read_csv(data_root / inputs["churn"])

    sample_n = getattr(cfg, "sample", None)
    if sample_n:
        churn_df = churn_df.head(sample_n)

    observed_col = getattr(cfg, "observed_col", "Churn")
    # Auto-detect observed column
    if observed_col not in churn_df.columns:
        for c in ["Churn", "churn", "churned", "target", "label"]:
            if c in churn_df.columns:
                observed_col = c
                break

    # Normalize target to 0/1
    churn_df[observed_col] = churn_df[observed_col].astype(str).str.strip().str.lower()
    bool_map = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "1.0": 1,
        "churn": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "0.0": 0,
        "no_churn": 0,
    }
    churn_df["target"] = churn_df[observed_col].map(bool_map)
    churn_df = churn_df.dropna(subset=["target"])
    churn_df["target"] = churn_df["target"].astype(int)

    # ── 2. Feature preparation ────────────────────────────────────────────────
    exclude = {observed_col, "target"}
    cat_cols = [
        c for c in churn_df.select_dtypes(include="object").columns if c not in exclude
    ]
    num_cols = [
        c for c in churn_df.select_dtypes(include="number").columns if c not in exclude
    ]

    X = churn_df[cat_cols + num_cols]
    y = churn_df["target"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )

    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y.values, test_size=0.2, random_state=settings.seed, stratify=y
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = fit_logistic_regression(X_train, y_train, seed=settings.seed)
    save_model(model, out_dir / "model.pkl")

    y_proba = model.predict_proba(X_test)[:, 1]
    clf_metrics = classification_metrics(y_test, y_proba)

    # ── 4. Threshold sweep ────────────────────────────────────────────────────
    step = float(getattr(cfg, "threshold_step", 0.01))
    eval_df = pd.DataFrame({"target": y_test, "p_churn": y_proba})
    thresholds_df = iterate_thresholds(eval_df, "target", "p_churn", step=step)

    # Cost-benefit
    cb = getattr(cfg, "cost_benefit", None) or {}
    benefit_tp = float(cb.get("benefit_tp", 1000) if isinstance(cb, dict) else 1000)
    benefit_tn = float(cb.get("benefit_tn", 0) if isinstance(cb, dict) else 0)
    cost_fp = float(cb.get("cost_fp", -250) if isinstance(cb, dict) else -250)
    cost_fn = float(cb.get("cost_fn", -500) if isinstance(cb, dict) else -500)

    thresholds_df["utility"] = (
        thresholds_df["Count_TP"] * benefit_tp
        + thresholds_df["Count_TN"] * benefit_tn
        + thresholds_df["Count_FP"] * cost_fp
        + thresholds_df["Count_FN"] * cost_fn
    )

    best_idx = thresholds_df["utility"].idxmax()
    best_threshold = float(thresholds_df.loc[best_idx, "Threshold"])
    best_utility = float(thresholds_df.loc[best_idx, "utility"])
    best_accuracy = float(thresholds_df.loc[best_idx, "Accuracy"])

    metrics = {
        **clf_metrics,
        "best_threshold": best_threshold,
        "best_utility": best_utility,
        "accuracy_at_best": best_accuracy,
    }

    # ── 5. Save ───────────────────────────────────────────────────────────────
    feat_out = pd.DataFrame(churn_df[["target", *num_cols[:5]]])
    write_parquet(feat_out, out_dir / "features.parquet")
    write_csv(thresholds_df, out_dir / "thresholds.csv")
    write_json(metrics, out_dir / "model_metrics.json")

    # Figure
    try:
        ptheme = plot_theme(title_size=16)
        fig = utility_by_threshold(
            thresholds_df["Threshold"].values,
            thresholds_df["utility"].values,
            title="Cost-Benefit Utility by Classification Threshold",
            theme=ptheme,
        )
        write_figure(fig, fig_dir / "utility_by_threshold.png")
    except Exception as e:
        logger.warning("Figure error: %s", e)

    logger.info(
        "Ch06 complete. ROC-AUC=%.4f Best threshold=%.2f Best utility=%.0f",
        clf_metrics["roc_auc"],
        best_threshold,
        best_utility,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ch06: Churn Prediction")
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
        logger.exception("Ch06 pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
