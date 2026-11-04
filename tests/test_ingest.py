"""Tests for schema validation and type normalization in ingest."""

from __future__ import annotations

import pandas as pd
import pytest

from costobs import ingest
from costobs.ingest import SchemaError, load_cur

_HEADER = (
    "usage_date,account_id,service,resource_id,"
    "owner,team,environment,unblended_cost,usage_amount\n"
)


def test_load_cur_types_and_blank_tags(tmp_path):
    csv = tmp_path / "mini.csv"
    csv.write_text(
        _HEADER
        + "2024-01-01,00111,AmazonEC2,r1,alice,platform,prod,10.5,100\n"
        + "2024-01-01,00111,AmazonS3,r2, ,,prod,2.0,5\n",
        encoding="utf-8",
    )

    df = load_cur(csv)

    assert str(df["usage_date"].dtype).startswith("datetime64")
    assert df["unblended_cost"].dtype == "float64"
    assert df["usage_amount"].dtype == "float64"

    # Blank / whitespace-only tags collapse to NA; real values survive.
    assert pd.isna(df.loc[1, "owner"])
    assert pd.isna(df.loc[1, "team"])
    assert df.loc[1, "environment"] == "prod"

    # account_id with leading zeros stays a string (not coerced to int).
    assert df["account_id"].tolist() == ["00111", "00111"]


def test_normalize_reorders_columns():
    raw = pd.DataFrame(
        {
            "unblended_cost": [1.0],
            "usage_amount": [1.0],
            "owner": ["a"],
            "team": ["t"],
            "environment": ["prod"],
            "resource_id": ["r"],
            "service": ["AmazonEC2"],
            "account_id": ["1"],
            "usage_date": ["2024-01-01"],
        }
    )
    out = ingest.normalize(raw)
    assert list(out.columns) == list(ingest.EXPECTED_COLUMNS)


def test_missing_column_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("usage_date,unblended_cost\n2024-01-01,1.0\n", encoding="utf-8")
    with pytest.raises(SchemaError):
        load_cur(csv)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cur(tmp_path / "nope.csv")
