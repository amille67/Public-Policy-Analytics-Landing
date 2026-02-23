"""Tests for mobility space-time panel interval aggregation."""

from __future__ import annotations

import pandas as pd

from national.loaders.mobility import build_spacetime_panel


def test_build_spacetime_panel_15min() -> None:
    pings = pd.DataFrame(
        {
            "GEOID": ["1001", "1001", "1001", "1003"],
            "timestamp": [
                "2024-01-01T08:01:00Z",
                "2024-01-01T08:10:00Z",
                "2024-01-01T08:20:00Z",
                "2024-01-01T08:05:00Z",
            ],
        }
    )

    result = build_spacetime_panel(pings, interval="15min")

    assert "ping_count" in result.columns
    # 1001: 08:00 bin has 2, 08:15 bin has 1; 1003: 08:00 has 1
    assert len(result) == 3
    assert int(result[result["GEOID"] == "1001"].iloc[0]["ping_count"]) == 2


def test_build_spacetime_panel_1h() -> None:
    pings = pd.DataFrame(
        {
            "GEOID": ["1001", "1001", "1001", "1003"],
            "timestamp": [
                "2024-01-01T08:01:00Z",
                "2024-01-01T08:10:00Z",
                "2024-01-01T08:20:00Z",
                "2024-01-01T08:05:00Z",
            ],
        }
    )

    result = build_spacetime_panel(pings, interval="1h")

    # each GEOID collapses to single hour-bin
    assert len(result) == 2
    assert int(result[result["GEOID"] == "1001"].iloc[0]["ping_count"]) == 3
