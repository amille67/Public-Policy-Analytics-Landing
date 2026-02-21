"""Unit tests for ppa.geo.nearest: mean_knn_distance."""

import numpy as np
import pytest

from ppa.geo.nearest import mean_knn_distance


class TestMeanKnnDistance:
    def test_known_result_k1(self) -> None:
        # measure_from = [[0,0]], measure_to = [[0,1],[0,2],[0,3]], k=1
        from_pts = np.array([[0.0, 0.0]])
        to_pts = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        result = mean_knn_distance(from_pts, to_pts, k=1)
        assert result.shape == (1,)
        assert abs(result[0] - 1.0) < 1e-9

    def test_known_result_k2(self) -> None:
        from_pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        to_pts = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        result = mean_knn_distance(from_pts, to_pts, k=2)
        assert result.shape == (2,)
        # from [0,0] to nearest 2: [0,1] dist=1, [0,2] dist=2 → mean=1.5
        assert abs(result[0] - 1.5) < 1e-9

    def test_output_shape(self) -> None:
        from_pts = np.random.rand(10, 2)
        to_pts = np.random.rand(5, 2)
        result = mean_knn_distance(from_pts, to_pts, k=3)
        assert result.shape == (10,)

    def test_raises_on_k_greater_than_m(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            mean_knn_distance(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), k=2)

    def test_raises_on_k_less_than_1(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            mean_knn_distance(
                np.array([[0.0, 0.0]]), np.array([[1.0, 1.0], [2.0, 2.0]]), k=0
            )

    def test_raises_on_nan_inputs_from(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            mean_knn_distance(
                np.array([[np.nan, 0.0]]), np.array([[1.0, 1.0]]), k=1
            )

    def test_raises_on_nan_inputs_to(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            mean_knn_distance(
                np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]]), k=1
            )

    def test_raises_on_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="mismatch"):
            mean_knn_distance(
                np.array([[0.0, 0.0]]),
                np.array([[1.0, 1.0, 1.0]]),
                k=1,
            )

    def test_k_equal_1_returns_nearest(self) -> None:
        from_pts = np.array([[0.0, 0.0]])
        to_pts = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = mean_knn_distance(from_pts, to_pts, k=1)
        # Distances: 5.0 and 1.0 → nearest is 1.0
        assert abs(result[0] - 1.0) < 1e-9

    def test_1d_inputs_promoted_to_2d(self) -> None:
        from_pts = [0.0, 1.0, 2.0]
        to_pts = [0.5, 1.5]
        result = mean_knn_distance(from_pts, to_pts, k=1)
        assert result.shape == (3,)
