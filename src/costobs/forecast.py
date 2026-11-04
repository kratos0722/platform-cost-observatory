"""Forecasting: project month-end spend per owner, deterministically.

Two explainable methods, both free of randomness:

``run_rate`` (default)
    Extrapolate month-to-date (MTD) spend at its average daily rate across the
    remaining days of the month: ``projected = MTD / days_elapsed * days_in_month``.

``linear``
    Fit a least-squares line to daily spend so far and sum observed days plus
    predicted (non-negative) future days. Captures within-month ramp/decline a
    flat run rate misses.

Each row carries the intermediate quantities and a plain-English
``explanation`` so the number is never a black box.
"""

from __future__ import annotations

import calendar
from typing import Optional, Union

import numpy as np
import pandas as pd

from .allocation import fill_unallocated

COST_COLUMN = "unblended_cost"
OWNER_COLUMN = "owner"

TimestampLike = Union[str, pd.Timestamp, "np.datetime64"]


def _resolve_as_of(df: pd.DataFrame, as_of: Optional[TimestampLike]) -> pd.Timestamp:
    if as_of is None:
        return pd.Timestamp(df["usage_date"].max()).normalize()
    return pd.Timestamp(as_of).normalize()


def _linear_projection(daily_costs: pd.Series, days_elapsed: int, days_in_month: int) -> float:
    """Least-squares projection of a single owner's month-end total."""
    x = np.arange(1, days_elapsed + 1, dtype="float64")
    y = daily_costs.reindex(range(1, days_elapsed + 1), fill_value=0.0).to_numpy(dtype="float64")
    observed = float(y.sum())

    if days_elapsed >= days_in_month:
        return round(observed, 6)
    if days_elapsed >= 2 and np.any(y > 0):
        slope, intercept = np.polyfit(x, y, 1)
    else:  # not enough signal to fit a line -> fall back to the mean level
        slope, intercept = 0.0, float(y.mean()) if len(y) else 0.0

    future_days = np.arange(days_elapsed + 1, days_in_month + 1, dtype="float64")
    future = np.clip(slope * future_days + intercept, 0.0, None).sum()
    return round(observed + float(future), 6)


def forecast_month_end(
    df: pd.DataFrame,
    as_of: Optional[TimestampLike] = None,
    method: str = "run_rate",
    cost_col: str = COST_COLUMN,
    owner_col: str = OWNER_COLUMN,
) -> pd.DataFrame:
    """Project end-of-month spend per owner for the month containing ``as_of``.

    Parameters
    ----------
    as_of:
        The "today" to forecast from. Defaults to the latest ``usage_date`` in
        the data. Only spend in the same calendar month, up to and including
        ``as_of``, feeds the projection.
    method:
        ``"run_rate"`` (default) or ``"linear"``.

    Returns a DataFrame (one row per owner) with ``mtd_cost``, ``days_elapsed``,
    ``days_in_month``, ``run_rate_per_day``, ``projected_month_end``, ``method``
    and a human-readable ``explanation``, sorted by projection descending.
    """
    if method not in {"run_rate", "linear"}:
        raise ValueError(f"unknown forecast method: {method!r}")

    work = df.copy()
    work["usage_date"] = pd.to_datetime(work["usage_date"]).dt.normalize()
    work = fill_unallocated(work, [owner_col])

    as_of_ts = _resolve_as_of(work, as_of)
    month_start = as_of_ts.replace(day=1)
    days_in_month = calendar.monthrange(as_of_ts.year, as_of_ts.month)[1]
    days_elapsed = int(as_of_ts.day)

    mtd = work[(work["usage_date"] >= month_start) & (work["usage_date"] <= as_of_ts)]

    rows: list[dict] = []
    for owner, group in mtd.groupby(owner_col, dropna=False):
        mtd_cost = float(group[cost_col].sum())
        run_rate = mtd_cost / days_elapsed if days_elapsed else 0.0

        if method == "run_rate":
            projected = round(run_rate * days_in_month, 6)
            explanation = (
                f"MTD ${mtd_cost:,.2f} over {days_elapsed} day(s) = "
                f"${run_rate:,.2f}/day x {days_in_month} days -> ${projected:,.2f}"
            )
        else:  # linear
            per_day = (
                group.assign(_dom=group["usage_date"].dt.day)
                .groupby("_dom")[cost_col]
                .sum()
            )
            projected = _linear_projection(per_day, days_elapsed, days_in_month)
            explanation = (
                f"least-squares trend on {days_elapsed} day(s) of spend "
                f"(MTD ${mtd_cost:,.2f}) projected to {days_in_month}-day month "
                f"-> ${projected:,.2f}"
            )

        rows.append(
            {
                owner_col: owner,
                "month": month_start.strftime("%Y-%m"),
                "as_of": as_of_ts.strftime("%Y-%m-%d"),
                "mtd_cost": round(mtd_cost, 6),
                "days_elapsed": days_elapsed,
                "days_in_month": days_in_month,
                "run_rate_per_day": round(run_rate, 6),
                "projected_month_end": projected,
                "method": method,
                "explanation": explanation,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["projected_month_end", owner_col], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
