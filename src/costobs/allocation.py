"""Cost allocation: clean groupby rollups with an explicit unallocated bucket.

Every aggregation routes untagged ``owner`` / ``team`` / ``environment`` values
through the sentinel :data:`UNALLOCATED` so that unattributable spend is never
silently dropped -- it shows up as its own line, which is exactly the number a
platform team wants to drive to zero.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

#: Bucket label used for spend whose allocation tag is missing / blank.
UNALLOCATED = "unallocated"

COST_COLUMN = "unblended_cost"

#: Tag columns whose missing values collapse into :data:`UNALLOCATED`.
_ALLOC_TAGS: frozenset[str] = frozenset({"owner", "team", "environment"})


def fill_unallocated(df: pd.DataFrame, dims: Sequence[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with missing allocation tags in ``dims`` filled.

    Only columns that are recognised allocation tags are touched; other
    grouping dimensions (``service``, ``usage_date``, ...) are left as-is.
    """
    out = df.copy()
    for dim in dims:
        if dim in _ALLOC_TAGS and dim in out.columns:
            cleaned = out[dim].astype("string").str.strip()
            out[dim] = cleaned.mask(cleaned.isna() | (cleaned.str.len() == 0), UNALLOCATED).astype(
                "string"
            )
    return out


def allocate(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    cost_col: str = COST_COLUMN,
) -> pd.DataFrame:
    """Sum ``cost_col`` grouped by ``group_cols``.

    Missing allocation tags are bucketed as :data:`UNALLOCATED`. The result is
    sorted by cost descending and has the cost column renamed to ``cost``.
    """
    cols = list(group_cols)
    filled = fill_unallocated(df, cols)
    grouped = (
        filled.groupby(cols, dropna=False)[cost_col]
        .sum()
        .reset_index()
        .rename(columns={cost_col: "cost"})
    )
    grouped["cost"] = grouped["cost"].round(6)
    sort_cols = ["cost", *cols]
    ascending = [False, *[True] * len(cols)]
    return grouped.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)


def by_owner(df: pd.DataFrame) -> pd.DataFrame:
    """Total cost per owner (untagged -> ``unallocated``)."""
    return allocate(df, ["owner"])


def by_team(df: pd.DataFrame) -> pd.DataFrame:
    """Total cost per team (untagged -> ``unallocated``)."""
    return allocate(df, ["team"])


def by_service(df: pd.DataFrame) -> pd.DataFrame:
    """Total cost per AWS service."""
    return allocate(df, ["service"])


def daily(df: pd.DataFrame) -> pd.DataFrame:
    """Total cost per calendar day."""
    return allocate(df, ["usage_date"])


def by_owner_day(df: pd.DataFrame) -> pd.DataFrame:
    """Total cost per owner per day -- the series forecasting/anomaly consume."""
    return allocate(df, ["owner", "usage_date"])
