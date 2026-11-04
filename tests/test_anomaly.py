"""Tests for spend-anomaly detection: catch spikes, ignore normal variation."""

from __future__ import annotations

import pandas as pd

from costobs import anomaly

# A smooth, realistically-noisy series with no genuine spike.
NORMAL = [100, 102, 98, 101, 99, 100, 103, 97, 100, 101,
          99, 102, 98, 100, 101, 99, 100, 102, 98, 101]


def _series_df(costs, owner: str = "x", start: str = "2024-02-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(costs), freq="D")
    return pd.DataFrame(
        {
            "usage_date": dates,
            "owner": [owner] * len(costs),
            "unblended_cost": [float(c) for c in costs],
        }
    )


def test_ignores_normal_variation():
    out = anomaly.detect_anomalies(_series_df(NORMAL))
    assert out.empty


def test_flat_series_never_flags():
    out = anomaly.detect_anomalies(_series_df([100.0] * 20))
    assert out.empty


def test_flags_injected_spike_with_reason():
    costs = list(NORMAL)
    costs[15] = 1000.0  # spike on 2024-02-16
    out = anomaly.detect_anomalies(_series_df(costs))

    assert len(out) == 1
    row = out.iloc[0]
    assert row["owner"] == "x"
    assert row["usage_date"] == pd.Timestamp("2024-02-16")
    assert row["cost"] == 1000.0
    assert row["threshold"] < row["cost"]
    assert "exceeded baseline" in row["reason"]
    assert row["zscore"] > 3.0


def test_spike_recovery_day_is_not_flagged():
    # The day after a spike must not itself be flagged just because its
    # baseline now includes the spike.
    costs = list(NORMAL)
    costs[15] = 1000.0
    out = anomaly.detect_anomalies(_series_df(costs))
    flagged_dates = set(out["usage_date"])
    assert pd.Timestamp("2024-02-17") not in flagged_dates


def test_day_over_day_method():
    costs = [100.0] * 10
    costs[5] = 400.0  # 4x jump on 2024-02-06
    out = anomaly.detect_anomalies(_series_df(costs), method="dod", dod_jump=3.0)
    assert (out["usage_date"] == pd.Timestamp("2024-02-06")).any()
    # The drop back down on the next day is not a spike.
    assert not (out["usage_date"] == pd.Timestamp("2024-02-07")).any()


def test_multi_column_grouping():
    # (x, EC2) carries the spike; (y, S3) is noisy-but-normal. Each entity's
    # series is scanned independently, so only one row should flag.
    x_costs = list(NORMAL)
    x_costs[15] = 1000.0
    y_costs = [c * 0.5 for c in NORMAL]
    df = pd.DataFrame(
        {
            "usage_date": list(pd.date_range("2024-02-01", periods=20, freq="D")) * 2,
            "owner": ["x"] * 20 + ["y"] * 20,
            "service": ["AmazonEC2"] * 20 + ["AmazonS3"] * 20,
            "unblended_cost": x_costs + y_costs,
        }
    )
    out = anomaly.detect_anomalies(df, group_cols=["owner", "service"])
    assert len(out) == 1
    assert out.iloc[0]["owner"] == "x"
    assert out.iloc[0]["service"] == "AmazonEC2"
