"""Unit tests for etl.create_high_value_views."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from etl.create_high_value_views import run_selected


def test_run_selected_builds_requested_views(monkeypatch: object, tmp_path: Path) -> None:
    written: list[Path] = []

    def _fake_runner() -> pd.DataFrame:
        return pd.DataFrame({"GEOID": ["47037010100"], "risk_score": [1.0]})

    from etl import create_high_value_views as mod

    # Updated to match final RUNNER_MAP (short keys)
    monkeypatch.setattr(mod, "RUNNER_MAP", {k: _fake_runner for k in mod.RUNNER_MAP})

    def _fake_write_parquet(df: pd.DataFrame, path: Path) -> None:
        written.append(path)

    monkeypatch.setattr(mod, "write_parquet", _fake_write_parquet)
    monkeypatch.setattr(mod, "write_geoparquet", _fake_write_parquet)

    config = tmp_path / "views_config.json"
    config.write_text(Path("etl/views_config.json").read_text(encoding="utf-8"), encoding="utf-8")

    out = run_selected("1,3", config_path=config)

    assert out == [1, 3]
    assert len(written) == 4


def test_config_must_define_exactly_seven_views(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"views": [{"view_id": 1, "runner_id": "01"}]}', encoding="utf-8")

    raised = False
    try:
        run_selected("all", config_path=bad)
    except ValueError:
        raised = True

    assert raised
