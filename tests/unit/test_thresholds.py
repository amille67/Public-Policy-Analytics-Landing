"""Unit tests for ppa.ml.thresholds: iterate_thresholds."""

import numpy as np
import pandas as pd
import pytest

from ppa.ml.thresholds import iterate_thresholds


def make_fixture_df() -> pd.DataFrame:
    """6-row fixture with known confusion at threshold 0.5."""
    return pd.DataFrame(
        {
            "observed": [1, 1, 1, 0, 0, 0],
            "prob": [0.9, 0.7, 0.4, 0.6, 0.3, 0.2],
        }
    )


class TestIterateThresholds:
    def test_threshold_count_default_step(self) -> None:
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        # Should have 100 rows: 0.01, 0.02, ..., 1.00
        assert len(result) == 100

    def test_threshold_column_present(self) -> None:
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        assert "Threshold" in result.columns

    def test_required_columns_present(self) -> None:
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        required = {
            "Count_TN", "Count_TP", "Count_FN", "Count_FP",
            "Rate_TP", "Rate_FP", "Rate_FN", "Rate_TN",
            "Accuracy", "Threshold",
        }
        assert required.issubset(set(result.columns))

    def test_strict_greater_than_behavior(self) -> None:
        """prob == threshold should classify as 0 (strict >)."""
        df = pd.DataFrame({"observed": [1, 0], "prob": [0.5, 0.5]})
        result = iterate_thresholds(df, "observed", "prob", step=0.5)
        # At threshold=0.5: prob=0.5 > 0.5 is False → pred=0 for both
        row = result[result["Threshold"] == 0.5].iloc[0]
        assert int(row["Count_TP"]) == 0
        assert int(row["Count_FP"]) == 0
        assert int(row["Count_TN"]) == 1  # obs=0, pred=0
        assert int(row["Count_FN"]) == 1  # obs=1, pred=0

    def test_fixture_counts_at_threshold_05(self) -> None:
        """At threshold 0.5, probs [0.9,0.7,0.6] > 0.5 → pred=1; [0.4,0.3,0.2] → pred=0.
        observed [1,1,1,0,0,0].
        TP=2 (obs=1,pred=1: prob=0.9,0.7), FN=1 (obs=1,pred=0: prob=0.4)
        FP=1 (obs=0,pred=1: prob=0.6), TN=2 (obs=0,pred=0: prob=0.3,0.2)
        """
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        row = result[result["Threshold"] == 0.50].iloc[0]
        assert int(row["Count_TP"]) == 2
        assert int(row["Count_FN"]) == 1
        assert int(row["Count_FP"]) == 1
        assert int(row["Count_TN"]) == 2

    def test_grouped_output_has_group_column(self) -> None:
        df = make_fixture_df()
        df["race"] = ["A", "A", "A", "B", "B", "B"]
        result = iterate_thresholds(df, "observed", "prob", group="race")
        assert "race" in result.columns

    def test_grouped_row_count(self) -> None:
        df = make_fixture_df()
        df["race"] = ["A", "A", "A", "B", "B", "B"]
        result = iterate_thresholds(df, "observed", "prob", group="race", step=0.1)
        # 10 thresholds * 2 groups = 20 rows
        assert len(result) == 20

    def test_accuracy_is_between_0_and_1(self) -> None:
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        valid_acc = result["Accuracy"].dropna()
        assert (valid_acc >= 0).all() and (valid_acc <= 1).all()

    def test_threshold_range(self) -> None:
        df = make_fixture_df()
        result = iterate_thresholds(df, "observed", "prob")
        assert result["Threshold"].min() == pytest.approx(0.01)
        assert result["Threshold"].max() == pytest.approx(1.00)
