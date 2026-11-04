#!/usr/bin/env python
"""Deterministic synthetic CUR-style data generator.

Emits a CSV under ``sample_data/`` (committed to the repo) so that tests and CI
never depend on regeneration. Everything is seeded; no network, no clock.

The generated data is entirely fictional (NDA-safe). It is shaped to exercise
every signal in the pipeline:

* a handful of resources are deliberately *untagged* (fully or partially), so
  tagging-hygiene coverage is well below 100%;
* one resource receives a large one-day cost *spike* on :data:`SPIKE_DATE`, so
  anomaly detection has a known target;
* the series spans ~60 days across 3 accounts and ends mid-month, so the
  month-end forecast is a genuine projection rather than a completed month.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 60
START_DATE = pd.Timestamp("2024-01-10")

# Known anomaly target (asserted by the integration test). The spike lands well
# inside February, which is fully covered, so the rolling baseline exists.
SPIKE_OWNER = "bob"
SPIKE_SERVICE = "AmazonEC2"
SPIKE_ACCOUNT = "111122223333"
SPIKE_DATE = pd.Timestamp("2024-02-20")
SPIKE_ADD_USD = 6000.0

# Per-service unit price used to derive a plausible usage_amount from cost.
UNIT_PRICE = {
    "AmazonEC2": 0.096,
    "AmazonS3": 0.023,
    "AmazonRDS": 0.145,
    "AWSLambda": 0.0000166667,
    "AmazonCloudWatch": 0.30,
}


@dataclass(frozen=True)
class Resource:
    account_id: str
    service: str
    owner: Optional[str]
    team: Optional[str]
    environment: Optional[str]
    base_cost: float


# A fixed fleet. ``None`` tags are deliberate hygiene gaps.
RESOURCES: tuple[Resource, ...] = (
    # --- fully tagged ---
    Resource("111122223333", "AmazonEC2", "bob", "platform", "prod", 120.0),  # spike target
    Resource("111122223333", "AmazonEC2", "alice", "platform", "prod", 90.0),
    Resource("111122223333", "AmazonS3", "alice", "data", "prod", 15.0),
    Resource("111122223333", "AmazonRDS", "carol", "data", "prod", 60.0),
    Resource("111122223333", "AWSLambda", "dave", "web", "staging", 8.0),
    Resource("222233334444", "AmazonEC2", "carol", "data", "prod", 75.0),
    Resource("222233334444", "AmazonS3", "bob", "platform", "dev", 10.0),
    Resource("222233334444", "AmazonCloudWatch", "dave", "web", "prod", 5.0),
    Resource("222233334444", "AmazonRDS", "alice", "data", "staging", 40.0),
    Resource("333344445555", "AmazonEC2", "dave", "web", "prod", 55.0),
    Resource("333344445555", "AWSLambda", "carol", "data", "dev", 6.0),
    Resource("333344445555", "AmazonS3", "bob", "platform", "prod", 12.0),
    # --- hygiene offenders ---
    Resource("111122223333", "AmazonEC2", None, None, None, 200.0),  # fully untagged, pricey
    Resource("222233334444", "AmazonRDS", None, None, None, 130.0),  # fully untagged
    Resource("333344445555", "AmazonEC2", "alice", None, "prod", 45.0),  # missing team
    Resource("111122223333", "AmazonS3", None, "data", "prod", 20.0),  # missing owner
    Resource("222233334444", "AWSLambda", "bob", "platform", None, 7.0),  # missing environment
)


def generate(seed: int = SEED, n_days: int = N_DAYS) -> pd.DataFrame:
    """Build the synthetic CUR frame deterministically."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")

    records: list[dict] = []
    for idx, res in enumerate(RESOURCES):
        resource_id = f"res-{idx:03d}-{res.service.lower()}"
        # One stable noise draw per (resource, day).
        noise = rng.normal(0.0, 0.08, size=n_days)
        for day_i, day in enumerate(dates):
            cost = res.base_cost * (1.0 + noise[day_i])
            cost = max(cost, 0.0)
            if (
                res.owner == SPIKE_OWNER
                and res.service == SPIKE_SERVICE
                and res.account_id == SPIKE_ACCOUNT
                and day == SPIKE_DATE
            ):
                cost += SPIKE_ADD_USD
            usage = cost / UNIT_PRICE[res.service]
            records.append(
                {
                    "usage_date": day.strftime("%Y-%m-%d"),
                    "account_id": res.account_id,
                    "service": res.service,
                    "resource_id": resource_id,
                    "owner": res.owner if res.owner is not None else "",
                    "team": res.team if res.team is not None else "",
                    "environment": res.environment if res.environment is not None else "",
                    "unblended_cost": round(float(cost), 6),
                    "usage_amount": round(float(usage), 4),
                }
            )

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["usage_date", "account_id", "resource_id"]).reset_index(drop=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic CUR-style sample data.")
    default_out = Path(__file__).resolve().parent.parent / "sample_data" / "cur_sample.csv"
    parser.add_argument("--out", type=Path, default=default_out, help="output CSV path")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--days", type=int, default=N_DAYS)
    args = parser.parse_args(argv)

    df = generate(seed=args.seed, n_days=args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")
    print(
        f"wrote {len(df):,} rows x {df.shape[1]} cols to {args.out} "
        f"(spike: {SPIKE_OWNER}/{SPIKE_SERVICE} on {SPIKE_DATE.date()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
