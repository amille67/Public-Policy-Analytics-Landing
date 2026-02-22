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


class SklearnDummyModel:
    """Model that exposes feature_names_in_ like a fitted sklearn estimator."""

    def __init__(self, probs: np.ndarray, feature_names: list[str]) -> None:
        self._probs = np.asarray(probs, dtype=float)
        self.feature_names_in_ = np.array(feature_names)

    def predict_proba(self, X: object) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else len(self._probs)
        p = self._probs[:n]
        return np.column_stack([1 - p, p])


def make_fixture(seed: int = 42) -> tuple[pd.DataFrame, DummyModel]:
    """8-row fixture with 2 races, feature columns, and fixed probabilities."""
    data = pd.DataFrame(
        {
            "race": ["African-American"] * 4 + ["Caucasian"] * 4,
            "Recidivated": [
                "Recidivate",
                "Recidivate",
                "notRecidivate",
                "notRecidivate",
                "Recidivate",
                "notRecidivate",
                "notRecidivate",
                "notRecidivate",
            ],
            # Feature columns are required: iterate_fairness infers features
            # by excluding observed_col and group_col when feature_cols=None.
            "age": [25, 30, 35, 40, 28, 33, 38, 45],
            "priors": [2, 1, 0, 3, 1, 0, 2, 4],
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
            "race",
            "True_Negative",
            "True_Positive",
            "False_Negative",
            "False_Positive",
            "False_Positive_Rate",
            "False_Negative_Rate",
            "Accuracy",
            "threshold",
        }
        assert required.issubset(set(result.columns))

    def test_uses_ge_rule(self) -> None:
        """p == threshold must be predicted positive (>=)."""
        # Need both groups present; threshold = 0.5 is NOT in seq(0.1,1,0.5)=[0.1,0.6]
        # Use threshold_by=0.1 and check at threshold pair "0.5, 0.5"
        data = pd.DataFrame(
            {
                "race": [
                    "African-American",
                    "African-American",
                    "Caucasian",
                    "Caucasian",
                ],
                "Recidivated": [
                    "Recidivate",
                    "notRecidivate",
                    "Recidivate",
                    "notRecidivate",
                ],
                "age": [25, 30, 35, 40],
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
                "age": [25, 30],
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
                "age": [25, 30],
            }
        )
        model = DummyModel(np.array([0.8, 0.2]))
        with pytest.raises(ValueError, match="unexpected labels"):
            iterate_fairness(data, model, threshold_by=0.5)

    # ── Bug 3: threshold_by validation ──────────────────────────────────────
    def test_raises_on_zero_threshold_by(self) -> None:
        data, model = make_fixture()
        with pytest.raises(ValueError, match="threshold_by must be > 0"):
            iterate_fairness(data, model, threshold_by=0)

    def test_raises_on_negative_threshold_by(self) -> None:
        data, model = make_fixture()
        with pytest.raises(ValueError, match="threshold_by must be > 0"):
            iterate_fairness(data, model, threshold_by=-0.1)

    # ── Bug 1: feature selection / label leakage prevention ─────────────────
    def test_feature_cols_none_excludes_outcome_and_group(self) -> None:
        """When feature_cols=None, model should not receive outcome/group cols."""
        calls: list[object] = []

        class RecordingModel:
            def predict_proba(self, X: object) -> np.ndarray:
                calls.append(X)
                n = len(X) if hasattr(X, "__len__") else 8
                p = np.array([0.8, 0.6, 0.4, 0.2, 0.7, 0.3, 0.25, 0.15])[:n]
                return np.column_stack([1 - p, p])

        data = pd.DataFrame(
            {
                "race": ["African-American"] * 4 + ["Caucasian"] * 4,
                "Recidivated": [
                    "Recidivate",
                    "Recidivate",
                    "notRecidivate",
                    "notRecidivate",
                    "Recidivate",
                    "notRecidivate",
                    "notRecidivate",
                    "notRecidivate",
                ],
                "age": [25, 30, 35, 40, 28, 33, 38, 45],
                "priors": [2, 1, 0, 3, 1, 0, 2, 4],
            }
        )
        model = RecordingModel()
        iterate_fairness(data, model, threshold_by=0.5, feature_cols=None)

        assert len(calls) == 1
        received = calls[0]
        # Should not contain outcome or group columns
        assert "Recidivated" not in received.columns  # type: ignore[union-attr]
        assert "race" not in received.columns  # type: ignore[union-attr]
        # Should contain the actual feature columns
        assert "age" in received.columns  # type: ignore[union-attr]
        assert "priors" in received.columns  # type: ignore[union-attr]

    def test_feature_names_in_takes_precedence_over_inference(self) -> None:
        """sklearn feature_names_in_ should be used instead of column inference."""
        data = pd.DataFrame(
            {
                "race": ["African-American"] * 4 + ["Caucasian"] * 4,
                "Recidivated": [
                    "Recidivate",
                    "Recidivate",
                    "notRecidivate",
                    "notRecidivate",
                    "Recidivate",
                    "notRecidivate",
                    "notRecidivate",
                    "notRecidivate",
                ],
                "age": [25, 30, 35, 40, 28, 33, 38, 45],
                "priors": [2, 1, 0, 3, 1, 0, 2, 4],
            }
        )
        probs = np.array([0.8, 0.6, 0.4, 0.2, 0.7, 0.3, 0.25, 0.15])
        model = SklearnDummyModel(probs, feature_names=["age", "priors"])
        # Should not raise even though we don't pass feature_cols
        result = iterate_fairness(data, model, threshold_by=0.5)
        assert len(result) > 0

    def test_raises_when_no_features_can_be_inferred(self) -> None:
        """If only outcome and group cols exist, raise a clear error."""
        data = pd.DataFrame(
            {
                "race": [
                    "African-American",
                    "African-American",
                    "Caucasian",
                    "Caucasian",
                ],
                "Recidivated": [
                    "Recidivate",
                    "notRecidivate",
                    "Recidivate",
                    "notRecidivate",
                ],
            }
        )
        model = DummyModel(np.array([0.8, 0.2, 0.7, 0.3]))
        with pytest.raises(ValueError, match="No feature columns remain"):
            iterate_fairness(data, model, threshold_by=0.5, feature_cols=None)
