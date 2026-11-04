"""Integration tests over the committed synthetic sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from costobs import ingest, report

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "cur_sample.csv"

# Must match the known spike injected by scripts/generate_sample.py.
SPIKE_OWNER = "bob"
SPIKE_DATE = "2024-02-20"


@pytest.fixture(scope="module")
def sample_df():
    return ingest.load_cur(SAMPLE)


def test_committed_sample_exists():
    assert SAMPLE.exists(), "sample_data/cur_sample.csv must be committed"


def test_report_structure_is_json_serializable(sample_df):
    rep = report.build_report(sample_df)
    assert set(rep) == {"meta", "tagging", "allocation", "forecast", "anomalies"}
    # Must round-trip through JSON without a custom encoder.
    json.loads(json.dumps(rep))


def test_sample_has_tagging_gaps(sample_df):
    cov = report.build_report(sample_df)["tagging"]["coverage"]
    assert 0 < cov["row_coverage_pct"] < 100
    assert cov["untagged_rows"] > 0
    assert cov["untagged_cost"] > 0


def test_sample_forecast_is_nonempty_and_sane(sample_df):
    fc = report.build_report(sample_df)["forecast"]
    assert len(fc) >= 1
    assert all(row["projected_month_end"] >= 0 for row in fc)
    assert all(row["days_elapsed"] <= row["days_in_month"] for row in fc)


def test_sample_anomaly_catches_known_spike(sample_df):
    anoms = report.build_report(sample_df)["anomalies"]
    assert len(anoms) >= 1
    owner_date_pairs = {(a["owner"], a["usage_date"]) for a in anoms}
    assert (SPIKE_OWNER, SPIKE_DATE) in owner_date_pairs


def test_render_text_includes_all_sections(sample_df):
    text = report.render_text(report.build_report(sample_df))
    for section in ("Tagging hygiene", "Cost allocation", "forecast", "Anomalies"):
        assert section in text
