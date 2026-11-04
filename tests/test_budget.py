"""Tests for budget-breach detection against the month-end forecast."""

from __future__ import annotations

import math

import pandas as pd

from costobs.budget import check_budgets


def _forecast(**owners: float) -> pd.DataFrame:
    """Build a minimal forecast frame: owner -> projected_month_end."""
    return pd.DataFrame(
        [{"owner": o, "projected_month_end": v} for o, v in owners.items()]
    )


def test_flags_owner_over_budget():
    fc = _forecast(alice=6000.0, bob=1000.0)
    out = check_budgets(fc, {"alice": 5000.0, "bob": 5000.0})
    by_owner = {r["owner"]: r for _, r in out.iterrows()}

    assert by_owner["alice"]["status"] == "over"
    assert by_owner["alice"]["over_under"] == 1000.0
    assert by_owner["alice"]["pct_of_budget"] == 120.0
    assert by_owner["bob"]["status"] == "under"
    assert by_owner["bob"]["over_under"] == -4000.0


def test_overspend_sorted_first():
    fc = _forecast(alice=6000.0, carol=9000.0, bob=1000.0)
    out = check_budgets(fc, {"alice": 5000.0, "bob": 5000.0, "carol": 5000.0})
    # carol (+4000) over alice (+1000) over bob (-4000)
    assert list(out["owner"]) == ["carol", "alice", "bob"]


def test_owner_without_budget_is_no_budget():
    fc = _forecast(alice=6000.0, dave=3000.0)
    out = check_budgets(fc, {"alice": 5000.0})
    dave = out[out["owner"] == "dave"].iloc[0]
    assert dave["status"] == "no_budget"
    assert math.isnan(dave["budget"])
    assert math.isnan(dave["over_under"])


def test_owner_matching_is_case_and_space_insensitive():
    fc = _forecast(Alice=6000.0)
    out = check_budgets(fc, {"  alice ": 5000.0})
    assert out.iloc[0]["status"] == "over"


def test_empty_forecast_returns_empty_frame():
    out = check_budgets(pd.DataFrame(), {"alice": 5000.0})
    assert out.empty
    assert "status" in out.columns
