"""Budget tracking: compare the month-end forecast against per-owner budgets.

Allocation and forecasting answer "what will this cost?"; budgets answer "is
that too much?". Given the forecast produced by :mod:`costobs.forecast` and a
mapping of ``owner -> monthly budget``, this flags owners whose *projected*
month-end spend breaches budget — early enough to act, rather than after the
invoice lands.
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

OWNER_COLUMN = "owner"
PROJECTION_COLUMN = "projected_month_end"


def check_budgets(
    forecast_df: pd.DataFrame,
    budgets: Mapping[str, float],
    owner_col: str = OWNER_COLUMN,
    projection_col: str = PROJECTION_COLUMN,
) -> pd.DataFrame:
    """Evaluate each forecast owner against its budget.

    Returns one row per owner in ``forecast_df`` with columns: owner,
    ``projected_month_end``, ``budget`` (NaN if none set), ``over_under``
    (projected − budget; positive means over), ``pct_of_budget`` and
    ``status`` (``"over"`` / ``"under"`` / ``"no_budget"``). Rows are sorted so
    the largest overspend comes first.
    """
    if forecast_df.empty:
        return pd.DataFrame(
            columns=[owner_col, projection_col, "budget", "over_under", "pct_of_budget", "status"]
        )

    # Normalize budget keys so lookup is not whitespace/case sensitive.
    normalized = {str(k).strip().lower(): float(v) for k, v in budgets.items()}

    rows: list[dict] = []
    for _, fc_row in forecast_df.iterrows():
        owner = fc_row[owner_col]
        projected = float(fc_row[projection_col])
        budget = normalized.get(str(owner).strip().lower())

        if budget is None:
            rows.append(
                {
                    owner_col: owner,
                    projection_col: round(projected, 6),
                    "budget": float("nan"),
                    "over_under": float("nan"),
                    "pct_of_budget": float("nan"),
                    "status": "no_budget",
                }
            )
            continue

        over_under = projected - budget
        pct = (projected / budget * 100.0) if budget > 0 else float("nan")
        rows.append(
            {
                owner_col: owner,
                projection_col: round(projected, 6),
                "budget": round(budget, 6),
                "over_under": round(over_under, 6),
                "pct_of_budget": round(pct, 2),
                "status": "over" if projected > budget else "under",
            }
        )

    result = pd.DataFrame(rows)
    # Owners with a budget first (by overspend desc), then the un-budgeted ones.
    result["_sort"] = result["over_under"].fillna(float("-inf"))
    result = result.sort_values(
        ["_sort", owner_col], ascending=[False, True], kind="stable"
    ).drop(columns="_sort")
    return result.reset_index(drop=True)
