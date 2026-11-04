"""Tests for cost-allocation rollups and the unallocated bucket."""

from __future__ import annotations

import pandas as pd

from costobs import allocation


def _fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "usage_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "account_id": ["a"] * 4,
            "service": ["AmazonEC2", "AmazonS3", "AmazonEC2", "AmazonS3"],
            "resource_id": ["r1", "r2", "r3", "r4"],
            "owner": ["alice", "alice", None, "bob"],
            "team": ["platform", "platform", "platform", "data"],
            "environment": ["prod"] * 4,
            "unblended_cost": [10.0, 5.0, 7.0, 3.0],
            "usage_amount": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_by_owner_sums_with_unallocated_bucket():
    out = allocation.by_owner(_fixture())
    totals = dict(zip(out["owner"], out["cost"]))

    assert totals["alice"] == 15.0
    assert totals["bob"] == 3.0
    assert totals["unallocated"] == 7.0  # the owner-less row
    assert round(out["cost"].sum(), 6) == 25.0


def test_by_service_totals():
    out = allocation.by_service(_fixture())
    totals = dict(zip(out["service"], out["cost"]))
    assert totals["AmazonEC2"] == 17.0
    assert totals["AmazonS3"] == 8.0


def test_multi_dimension_allocation():
    out = allocation.allocate(_fixture(), ["owner", "usage_date"])
    # (alice, 1/1)=15, (unallocated, 1/2)=7, (bob, 1/2)=3
    assert len(out) == 3
    alice_jan1 = out[
        (out["owner"] == "alice") & (out["usage_date"] == pd.Timestamp("2024-01-01"))
    ]
    assert alice_jan1.iloc[0]["cost"] == 15.0


def test_result_sorted_descending_by_cost():
    out = allocation.by_owner(_fixture())
    assert out["cost"].tolist() == sorted(out["cost"].tolist(), reverse=True)
