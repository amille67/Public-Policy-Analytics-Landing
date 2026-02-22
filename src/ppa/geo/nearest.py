"""Mean k-nearest-neighbor distance computation (Python equivalent of nn_function)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_knn_distance(
    measure_from: ArrayLike,
    measure_to: ArrayLike,
    k: int,
) -> np.ndarray:
    """Compute the mean distance to the k nearest neighbors.

    This is the Python equivalent of the R ``nn_function`` helper.
    For each row in ``measure_from``, computes the mean distance to the
    ``k`` nearest points in ``measure_to``.

    Args:
        measure_from: Array-like of shape (n, d) — query points.
        measure_to: Array-like of shape (m, d) — reference points.
        k: Number of nearest neighbors to average; must satisfy 1 <= k <= m.

    Returns:
        np.ndarray of shape (n,) containing the mean k-NN distance for each
        query point. Units match those of the input coordinates (pipelines
        must ensure meters for real-world distances).

    Raises:
        ValueError: If k < 1, k > m, arrays have mismatched dimensions,
            or any NaN coordinates are present.
    """
    from sklearn.neighbors import NearestNeighbors

    from_arr = np.asarray(measure_from, dtype=float)
    to_arr = np.asarray(measure_to, dtype=float)

    if from_arr.ndim == 1:
        from_arr = from_arr.reshape(-1, 1)
    if to_arr.ndim == 1:
        to_arr = to_arr.reshape(-1, 1)

    if from_arr.ndim != 2 or to_arr.ndim != 2:
        raise ValueError("measure_from and measure_to must be 2-D arrays")
    if from_arr.shape[1] != to_arr.shape[1]:
        raise ValueError(
            f"Dimension mismatch: measure_from has {from_arr.shape[1]} dims, "
            f"measure_to has {to_arr.shape[1]} dims"
        )
    if np.any(np.isnan(from_arr)):
        raise ValueError("measure_from contains NaN values")
    if np.any(np.isnan(to_arr)):
        raise ValueError("measure_to contains NaN values")

    m = len(to_arr)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > m:
        raise ValueError(f"k ({k}) cannot exceed the number of reference points ({m})")

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(to_arr)
    distances, _ = nn.kneighbors(from_arr)

    return distances.mean(axis=1)  # type: ignore[no-any-return]
