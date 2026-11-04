"""Tests for tagging-hygiene detection and coverage math."""

from __future__ import annotations

import pandas as pd

from costobs import tagging


def _fixture() -> pd.DataFrame:
    """Four rows: two fully tagged, one missing owner, one missing environment."""
    return pd.DataFrame(
        {
            "usage_date": pd.to_datetime(["2024-01-01"] * 4),
            "account_id": ["a"] * 4,
            "service": ["AmazonEC2", "AmazonS3", "AmazonEC2", "AmazonRDS"],
            "resource_id": ["r1", "r2", "r3", "r4"],
            "owner": ["alice", "bob", None, "carol"],
            "team": ["platform", "data", "data", "data"],
            "environment": ["prod", "prod", "prod", None],
            "unblended_cost": [100.0, 50.0, 40.0, 10.0],
            "usage_amount": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_coverage_counts_and_percentages():
    cov = tagging.tag_coverage(_fixture())

    assert cov["total_rows"] == 4
    assert cov["tagged_rows"] == 2
    assert cov["untagged_rows"] == 2
    assert cov["row_coverage_pct"] == 50.0

    # Per-tag presence across all rows.
    assert cov["per_tag_coverage_pct"]["owner"] == 75.0
    assert cov["per_tag_coverage_pct"]["team"] == 100.0
    assert cov["per_tag_coverage_pct"]["environment"] == 75.0

    # Cost-weighted coverage: tagged = 100 + 50 of 200 total.
    assert cov["total_cost"] == 200.0
    assert cov["tagged_cost"] == 150.0
    assert cov["untagged_cost"] == 50.0
    assert cov["cost_coverage_pct"] == 75.0


def test_flags_identify_exact_missing_tags():
    flagged = tagging.add_tag_flags(_fixture())

    assert flagged.loc[0, "missing_tags"] == []
    assert flagged.loc[2, "missing_tags"] == ["owner"]
    assert flagged.loc[3, "missing_tags"] == ["environment"]
    assert flagged["is_fully_tagged"].tolist() == [True, True, False, False]
    assert flagged["n_missing_tags"].tolist() == [0, 0, 1, 1]


def test_worst_offenders_rank_untagged_spend_by_service():
    offenders = tagging.worst_offenders(_fixture(), by="service")

    # Only untagged rows contribute: EC2 ($40) and RDS ($10).
    assert set(offenders["service"]) == {"AmazonEC2", "AmazonRDS"}
    assert offenders.iloc[0]["service"] == "AmazonEC2"
    assert offenders.iloc[0]["untagged_cost"] == 40.0
    assert offenders.iloc[0]["untagged_rows"] == 1


def test_fully_tagged_input_has_no_offenders():
    df = _fixture()
    df.loc[2, "owner"] = "dave"
    df.loc[3, "environment"] = "prod"
    cov = tagging.tag_coverage(df)
    assert cov["row_coverage_pct"] == 100.0
    assert tagging.worst_offenders(df).empty
