"""Unit tests for ppa.raster.convert: rast_to_df."""

import math

import numpy as np
import pytest


def make_in_memory_raster(
    data: np.ndarray,
    nodata: float | None = None,
    transform=None,
):
    """Create an in-memory rasterio dataset from a numpy array."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    rows, cols = data.shape
    if transform is None:
        transform = from_bounds(0, 0, cols, rows, cols, rows)

    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": cols,
        "height": rows,
        "count": 1,
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    mem = MemoryFile()
    with mem.open(**profile) as ds:
        ds.write(data, 1)
    return mem.open()


class TestRastToDf:
    def test_shape_and_columns(self) -> None:
        try:
            from ppa.raster.convert import rast_to_df
        except ImportError:
            pytest.skip("rasterio not installed")

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ds = make_in_memory_raster(data)
        df = rast_to_df(ds)

        assert set(df.columns) == {"x", "y", "value"}
        assert len(df) == 6  # 2 rows * 3 cols

    def test_nodata_to_nan(self) -> None:
        try:
            from ppa.raster.convert import rast_to_df
        except ImportError:
            pytest.skip("rasterio not installed")

        data = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
        ds = make_in_memory_raster(data, nodata=-9999.0)
        df = rast_to_df(ds)

        nan_count = df["value"].isna().sum()
        assert nan_count == 1

    def test_coordinate_centers(self) -> None:
        try:
            import rasterio
            from rasterio.transform import from_origin
            from ppa.raster.convert import rast_to_df
        except ImportError:
            pytest.skip("rasterio not installed")

        # 1x1 raster at origin; cell size 10
        data = np.array([[42.0]], dtype=np.float32)
        transform = from_origin(0, 10, 10, 10)  # west=0, north=10, xsize=10, ysize=10
        ds = make_in_memory_raster(data, transform=transform)
        df = rast_to_df(ds)

        # Cell center should be at (5, 5)
        assert abs(df["x"].iloc[0] - 5.0) < 1e-6
        assert abs(df["y"].iloc[0] - 5.0) < 1e-6

    def test_invalid_band_raises(self) -> None:
        try:
            from ppa.raster.convert import rast_to_df
        except ImportError:
            pytest.skip("rasterio not installed")

        data = np.array([[1.0, 2.0]], dtype=np.float32)
        ds = make_in_memory_raster(data)
        with pytest.raises(ValueError):
            rast_to_df(ds, band=99)

    def test_all_nodata_to_nan(self) -> None:
        try:
            from ppa.raster.convert import rast_to_df
        except ImportError:
            pytest.skip("rasterio not installed")

        data = np.full((3, 3), -1.0, dtype=np.float32)
        ds = make_in_memory_raster(data, nodata=-1.0)
        df = rast_to_df(ds)
        assert df["value"].isna().all()
