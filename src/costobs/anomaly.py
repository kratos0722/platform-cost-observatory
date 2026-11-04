"""Anomaly detection: flag per-entity spend spikes with a clear reason.

The daily cost series is built per group (``owner`` by default, but any set of
columns works, e.g. ``["owner", "service"]``) over a gap-free date range (days
with no spend count as ``$0``). Two methods are offered:

``rolling_z`` (default)
    Flag a day whose cost exceeds ``baseline_mean + k * baseline_std`` where the
    baseline is the *prior* ``window`` days (the current day is excluded from
    its own baseline). A perfectly flat history has ``std == 0`` and therefore
    never fires. A second gate, ``min_rel_increase``, additionally requires the
    day to sit a minimum fraction above the baseline mean, so that a very smooth
    (tiny-std) series does not page on sub-percent wobble -- only a genuine
    spike clears both gates, while ordinary noise is ignored.

``dod``
    Day-over-day jump: flag when today is at least ``dod_jump`` times yesterday.

Every flagged row includes a ``reason`` string stating the numbers behind the
call, so an on-call engineer can act without rerunning the math.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .allocation import fill_unallocated

COST_COLUMN = "unblended_cost"

_RESULT_COLUMNS = [
    "usage_date",
    "cost",
    "baseline_mean",
    "baseline_std",
    "threshold",
    "zscore",
    "method",
    "reason",
]


def _as_key_tuple(keys) -> tuple:
    return keys if isinstance(keys, tuple) else (keys,)


def detect_anomalies(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("owner",),
    k: float = 3.0,
    window: int = 7,
    min_periods: int = 3,
    method: str = "rolling_z",
    cost_col: str = COST_COLUMN,
    dod_jump: float = 3.0,
    min_dollar: float = 1.0,
    min_rel_increase: float = 0.25,
) -> pd.DataFrame:
    """Return one row per flagged (group, day) spend anomaly.

    Parameters
    ----------
    group_cols:
        Columns defining an entity whose daily series is scanned independently.
    k:
        Standard-deviation multiplier for the ``rolling_z`` threshold.
    window, min_periods:
        Rolling baseline length and the minimum prior days required before any
        day can be flagged.
    method:
        ``"rolling_z"`` (default) or ``"dod"``.
    min_dollar:
        Absolute floor; sub-``min_dollar`` costs are never flagged (keeps noise
        on near-zero series out of the report).
    min_rel_increase:
        For ``rolling_z``, the day must also be at least this fraction above the
        baseline mean (e.g. ``0.25`` = 25%). Suppresses statistically-"unusual"
        but economically-trivial wobble on very smooth series.

    Result columns: the group columns, ``usage_date``, ``cost``,
    ``baseline_mean``, ``baseline_std``, ``threshold``, ``zscore``, ``method``
    and ``reason``.
    """
    if method not in {"rolling_z", "dod"}:
        raise ValueError(f"unknown anomaly method: {method!r}")

    cols = list(group_cols)
    work = df.copy()
    work["usage_date"] = pd.to_datetime(work["usage_date"]).dt.normalize()
    work = fill_unallocated(work, cols)

    daily = work.groupby([*cols, "usage_date"], dropna=False)[cost_col].sum().reset_index()
    if daily.empty:
        return pd.DataFrame(columns=[*cols, *_RESULT_COLUMNS])

    full_range = pd.date_range(daily["usage_date"].min(), daily["usage_date"].max(), freq="D")

    records: list[dict] = []
    for keys, grp in daily.groupby(cols, dropna=False):
        key_tuple = _as_key_tuple(keys)
        key_map = dict(zip(cols, key_tuple))
        label = ", ".join(f"{c}={key_map[c]}" for c in cols)

        series = (
            grp.set_index("usage_date")[cost_col]
            .reindex(full_range, fill_value=0.0)
            .astype("float64")
        )

        if method == "rolling_z":
            baseline = series.shift(1)
            mean = baseline.rolling(window, min_periods=min_periods).mean()
            std = baseline.rolling(window, min_periods=min_periods).std(ddof=0)
            threshold = mean + k * std
            flagged = (
                (std > 0)
                & (series > threshold)
                & (series >= mean * (1.0 + min_rel_increase))
                & (series >= min_dollar)
            )

            for date in series.index[flagged.fillna(False)]:
                cost = float(series.loc[date])
                mu = float(mean.loc[date])
                sigma = float(std.loc[date])
                thr = float(threshold.loc[date])
                z = (cost - mu) / sigma if sigma else float("inf")
                records.append(
                    {
                        **key_map,
                        "usage_date": date,
                        "cost": round(cost, 6),
                        "baseline_mean": round(mu, 6),
                        "baseline_std": round(sigma, 6),
                        "threshold": round(thr, 6),
                        "zscore": round(z, 4),
                        "method": method,
                        "reason": (
                            f"{label} spend ${cost:,.2f} on "
                            f"{date.strftime('%Y-%m-%d')} exceeded baseline "
                            f"${mu:,.2f} + {k:g}*${sigma:,.2f} = ${thr:,.2f} "
                            f"(z={z:.1f})"
                        ),
                    }
                )
        else:  # dod
            prev = series.shift(1)
            flagged = (prev > 0) & (series >= prev * dod_jump) & (series >= min_dollar)
            for date in series.index[flagged.fillna(False)]:
                cost = float(series.loc[date])
                prior = float(prev.loc[date])
                ratio = cost / prior if prior else float("inf")
                records.append(
                    {
                        **key_map,
                        "usage_date": date,
                        "cost": round(cost, 6),
                        "baseline_mean": round(prior, 6),
                        "baseline_std": 0.0,
                        "threshold": round(prior * dod_jump, 6),
                        "zscore": round(ratio, 4),
                        "method": method,
                        "reason": (
                            f"{label} spend ${cost:,.2f} on "
                            f"{date.strftime('%Y-%m-%d')} was {ratio:.1f}x the "
                            f"prior day's ${prior:,.2f} (>= {dod_jump:g}x)"
                        ),
                    }
                )

    if not records:
        return pd.DataFrame(columns=[*cols, *_RESULT_COLUMNS])

    result = pd.DataFrame.from_records(records)
    return result.sort_values([*cols, "usage_date"], kind="stable").reset_index(drop=True)
