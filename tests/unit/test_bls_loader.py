"""Tests for BLS loader API key injection behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from national.loaders.bls import _fetch_qcew_raw, fetch_qcew


def test_fetch_qcew_injects_api_key_from_env() -> None:
    csv = "own_code,industry_code,annual_avg_emplvl\n5,44,100\n"

    # Patch the cached raw function directly (bypasses joblib cache layer)
    with (
        patch("national.loaders.bls._fetch_qcew_raw", return_value=csv) as mock_raw,
        patch.dict("os.environ", {"BLS_API_KEY": "abc123"}),
    ):
        fetch_qcew(2022, "42101")

    assert mock_raw.called
    # _fetch_qcew_raw(year, quarter, area_fips, registration_key)
    call = mock_raw.call_args
    registration_key = (
        call.kwargs.get("registration_key") if call.kwargs else call.args[3]
    )
    assert registration_key == "abc123"


def test_fetch_qcew_raw_accepts_explicit_key() -> None:
    mock_resp = MagicMock()
    mock_resp.text = "a,b\n1,2\n"
    mock_resp.raise_for_status.return_value = None
    # Use .func to bypass the joblib cache and call the underlying function
    with patch("national.loaders.bls.requests.get", return_value=mock_resp) as mock_get:
        _fetch_qcew_raw.func(2022, "a", "42101", "xyz")
    assert mock_get.call_args.kwargs["params"]["registrationkey"] == "xyz"
