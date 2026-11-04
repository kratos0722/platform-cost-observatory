"""Tests for deterministic month-end forecasting."""

from __future__ import annotations

import pandas as pd

from costobs import forecast


def _owner_series(owner: str, costs: list[float], start: str = "2024-04-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(costs), freq="D")
    return pd.DataFrame(
        {"usage_date": dates, "owner": [owner] * len(costs), "unblended_cost": costs}
    )


def test_run_rate_projection_matches_hand_calc():
    # $10/day for 10 days of a 30-day month, as_of day 10.
    df = _owner_series("alice", [10.0] * 10)
    out = forecast.forecast_month_end(df, as_of="2024-04-10", method="run_rate")

    row = out.iloc[0]
    assert row["owner"] == "alice"
    assert row["mtd_cost"] == 100.0
    assert row["days_elapsed"] == 10
    assert row["days_in_month"] == 30
    assert row["run_rate_per_day"] == 10.0
    assert row["projected_month_end"] == 300.0  # 10/day * 30 days
    assert "MTD" in row["explanation"] and "300.00" in row["explanation"]


def test_multiple_owners_default_as_of_and_ordering():
    df = pd.concat(
        [_owner_series("alice", [10.0] * 5), _owner_series("bob", [20.0] * 5)],
        ignore_index=True,
    )
    # Default as_of = latest usage_date = 2024-04-05 (5 of 30 days).
    out = forecast.forecast_month_end(df)
    projected = dict(zip(out["owner"], out["projected_month_end"]))

    assert projected["alice"] == 300.0  # 50 / 5 * 30
    assert projected["bob"] == 600.0  # 100 / 5 * 30
    assert out.iloc[0]["owner"] == "bob"  # sorted by projection desc


def test_untagged_owner_forecast_as_unallocated():
    df = _owner_series("alice", [10.0] * 5)
    df.loc[df.index[:2], "owner"] = None
    out = forecast.forecast_month_end(df, as_of="2024-04-05")
    assert "unallocated" in set(out["owner"])


def test_linear_method_on_flat_series_matches_run_rate():
    df = _owner_series("alice", [10.0] * 10)
    out = forecast.forecast_month_end(df, as_of="2024-04-10", method="linear")
    row = out.iloc[0]
    assert row["method"] == "linear"
    # A flat history extrapolates to the same ~$300 as the run rate.
    assert abs(row["projected_month_end"] - 300.0) < 1e-3
