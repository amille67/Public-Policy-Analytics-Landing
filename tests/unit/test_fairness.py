"""Unit tests for ppa.ml.fairness: iterate_fairness."""

import numpy as np
import pandas as pd
import pytest

from ppa.ml.fairness import iterate_fairness


class DummyModel:
    """Model returning fixed probability vector."""

    def __init__(self, probs: np.ndarray) -> None:
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X: object) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else len(self._probs)
        p = self._probs[:n]
        return np.column_stack([1 - p, p])


def make_fixture(seed: int = 42) -> tuple[pd.DataFrame, DummyModel]:
    """8-row fixture with 2 races and fixed probabilities."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "race": ["African-American"] * 4 + ["Caucasian"] * 4,
            "Recidivated": [
                "Recidivate", "Recidivate", "notRecidivate", "notRecidivate",
                "Recidivate", "notRecidivate", "notRecidivate", "notRecidivate",
            ],
        }
    )
    probs = np.array([0.8, 0.6, 0.4, 0.2, 0.7, 0.3, 0.25, 0.15])
    return data, DummyModel(probs)


class TestIterateFairness:
    def test_grid_rowcount_default(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        # 10 thresholds * 10 thresholds * 2 groups = 200 rows
        assert len(result) == 200

    def test_grid_rowcount_coarser(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.5)
        # seq(0.1, 1.0, 0.5) in R → [0.1, 0.6] = 2 thresholds
        # 2 * 2 * 2 groups = 8 rows
        assert len(result) == 8

    def test_threshold_column_present(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        assert "threshold" in result.columns

    def test_required_columns_present(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        required = {
            "race", "True_Negative", "True_Positive",
            "False_Negative", "False_Positive",
            "False_Positive_Rate", "False_Negative_Rate",
            "Accuracy", "threshold",
        }
        assert required.issubset(set(result.columns))

    def test_uses_ge_rule(self) -> None:
        """p == threshold must be predicted positive (>=)."""
        # Need both groups present; threshold = 0.5 is NOT in seq(0.1,1,0.5)=[0.1,0.6]
        # Use threshold_by=0.1 and check at threshold pair "0.5, 0.5"
        data = pd.DataFrame(
            {
                "race": ["African-American", "African-American", "Caucasian", "Caucasian"],
                "Recidivated": ["Recidivate", "notRecidivate", "Recidivate", "notRecidivate"],
            }
        )
        # All probs = 0.5
        model = DummyModel(np.array([0.5, 0.5, 0.5, 0.5]))
        result = iterate_fairness(data, model, threshold_by=0.1)
        # At threshold pair "0.5, 0.5": prob=0.5 >= 0.5 → predicted positive
        row = result[
            (result["threshold"] == "0.5, 0.5") & (result["race"] == "African-American")
        ]
        assert not row.empty, "Expected row for threshold 0.5, 0.5"
        # Both AA rows: prob=0.5 >= t=0.5 → predicted positive for both
        # AA: obs=[Recidivate, notRecidivate], pred=[Recidivate, Recidivate]
        # TP=1, FP=1
        assert int(row["True_Positive"].iloc[0]) == 1
        assert int(row["False_Positive"].iloc[0]) == 1

    def test_raises_if_group_missing(self) -> None:
        data = pd.DataFrame(
            {
                "race": ["African-American", "African-American"],
                "Recidivated": ["Recidivate", "notRecidivate"],
            }
        )
        model = DummyModel(np.array([0.8, 0.2]))
        with pytest.raises(ValueError, match="Caucasian"):
            iterate_fairness(data, model, threshold_by=0.5)

    def test_both_groups_present_in_output(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        groups = set(result["race"].unique())
        assert "African-American" in groups
        assert "Caucasian" in groups

    def test_accuracy_between_0_and_1(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        valid = result["Accuracy"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_fpr_fnr_between_0_and_1(self) -> None:
        data, model = make_fixture()
        result = iterate_fairness(data, model, threshold_by=0.1)
        assert (result["False_Positive_Rate"].dropna().between(0, 1)).all()
        assert (result["False_Negative_Rate"].dropna().between(0, 1)).all()

    def test_raises_on_invalid_observed_labels(self) -> None:
        data = pd.DataFrame(
            {
                "race": ["African-American", "Caucasian"],
                "Recidivated": ["yes", "no"],
            }
        )
        model = DummyModel(np.array([0.8, 0.2]))
        with pytest.raises(ValueError, match="unexpected labels"):
            iterate_fairness(data, model, threshold_by=0.5)
