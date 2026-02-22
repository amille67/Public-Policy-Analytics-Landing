"""Unit tests for ppa.stats.quantiles: q5 and qbr."""

import numpy as np
import pandas as pd

from ppa.stats.quantiles import q5, qbr


class TestQ5:
    def test_q5_100_values_even_bins(self) -> None:
        values = list(range(1, 101))
        result = q5(values)
        counts = pd.Series(result).value_counts().sort_index()
        assert len(counts) == 5
        assert all(counts == 20), f"Expected 20 per bin, got {counts.tolist()}"

    def test_q5_returns_categorical(self) -> None:
        result = q5([1, 2, 3, 4, 5])
        assert isinstance(result, pd.Categorical)

    def test_q5_categories_always_1_to_5(self) -> None:
        result = q5([1, 2, 3])
        assert list(result.categories) == [1, 2, 3, 4, 5]

    def test_q5_preserves_nan_positions(self) -> None:
        values = [1, np.nan, 2, np.nan, 3, 4, 5, 6, 7, 8]
        result = q5(values)
        result_series = pd.Series(result)
        assert pd.isna(result_series.iloc[1])
        assert pd.isna(result_series.iloc[3])

    def test_q5_all_nan(self) -> None:
        result = q5([np.nan, np.nan, np.nan])
        result_series = pd.Series(result)
        assert result_series.isna().all()

    def test_q5_small_sample_less_than_5(self) -> None:
        result = q5([10, 20, 30])
        result_series = pd.Series(result).dropna()
        assert len(result_series) == 3
        assert result_series.notna().all()

    def test_q5_ties_deterministic(self) -> None:
        values = [1, 1, 1, 2, 2, 3, 3, 3, 4, 5]
        r1 = q5(values)
        r2 = q5(values)
        assert list(r1) == list(r2)

    def test_q5_10_values(self) -> None:
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = pd.Series(q5(values)).astype(float)
        assert result.min() == 1.0
        assert result.max() == 5.0

    def test_q5_ascending_order(self) -> None:
        values = list(range(50))
        result = pd.Series(q5(values)).astype(float)
        # First elements should be tile 1
        assert result.iloc[0] == 1.0
        # Last element should be tile 5
        assert result.iloc[-1] == 5.0


class TestQbr:
    def test_qbr_returns_list_of_5(self) -> None:
        df = pd.DataFrame({"v": range(100)})
        result = qbr(df, "v")
        assert isinstance(result, list)
        assert len(result) == 5

    def test_qbr_default_rounding(self) -> None:
        df = pd.DataFrame({"v": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
        result = qbr(df, "v", rnd=None)
        # Each element should be a string
        assert all(isinstance(s, str) for s in result)

    def test_qbr_rnd_false_formats_3_decimals(self) -> None:
        df = pd.DataFrame({"v": list(range(100))})
        result = qbr(df, "v", rnd=False)
        # Format should be x.xxx
        for s in result:
            assert "." in s
            decimal_part = s.split(".")[-1]
            assert len(decimal_part) == 3

    def test_qbr_all_nan_returns_nan_strings(self) -> None:
        df = pd.DataFrame({"v": [np.nan, np.nan, np.nan]})
        result = qbr(df, "v")
        assert result == ["nan"] * 5

    def test_qbr_known_quantiles_default(self) -> None:
        # Values 0..9; round(0..9,0) = same; q at [.01,.2,.4,.6,.8]
        # q(.01) of [0..9] ≈ 0.09; formatted with {:g} → "0.09"
        df = pd.DataFrame({"v": list(range(10))})
        result = qbr(df, "v", rnd=None)
        # All results should be string-castable to float
        float_vals = [float(s) for s in result]
        # Quantile at 0.01 should be near 0
        assert float_vals[0] < 1.0
        # Quantile at 0.8 should be near 7
        assert float_vals[-1] > 6.0

    def test_qbr_fixture_specific_values(self) -> None:
        df = pd.DataFrame({"v": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]})
        result = qbr(df, "v", rnd=None)
        # All values should be numeric strings
        for s in result:
            float(s)  # Should not raise
