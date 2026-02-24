"""MVP contract tests for Nashville view materialization counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl.create_high_value_views import run_selected


def test_views_match_expected_counts(monkeypatch: object, tmp_path: Path) -> None:
    from etl import create_high_value_views as mod

    def _fake_runner() -> pd.DataFrame:
        return pd.DataFrame({"GEOID": ["47037010100", "47037010200"], "metric": [1.0, 3.0]})

    monkeypatch.setattr(mod, "RUNNER_MAP", {k: _fake_runner for k in mod.RUNNER_MAP})

    written: list[Path] = []

    def _fake_write(df: pd.DataFrame, path: Path) -> None:
        written.append(path)

    monkeypatch.setattr(mod, "write_parquet", _fake_write)
    monkeypatch.setattr(mod, "write_geoparquet", _fake_write)

    config = tmp_path / "views_config.json"
    config.write_text(Path("etl/views_config.json").read_text(encoding="utf-8"), encoding="utf-8")

    out = run_selected("all", config_path=config)

    assert out == [1, 2, 3, 4, 5, 6, 7]
    assert len(written) == 14
