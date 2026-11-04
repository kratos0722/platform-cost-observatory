"""Tagging hygiene: which resources lack required allocation tags.

Allocation tags (``owner``, ``team``, ``environment`` by default) are what let
finance attribute spend back to a team. Missing tags are therefore both a
governance problem and a driver of "unallocated" cost. This module answers:

* which line items are fully tagged (:func:`add_tag_flags`),
* what fraction of rows / dollars are covered (:func:`tag_coverage`), and
* where the untagged money is concentrated (:func:`worst_offenders`).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import pandas as pd

from .ingest import REQUIRED_TAGS

COST_COLUMN = "unblended_cost"


def _is_present(series: pd.Series) -> pd.Series:
    """Boolean mask: ``True`` where a tag value is non-null and non-blank."""
    cleaned = series.astype("string").str.strip()
    return cleaned.notna() & (cleaned.str.len() > 0)


def add_tag_flags(df: pd.DataFrame, required_tags: Iterable[str] = REQUIRED_TAGS) -> pd.DataFrame:
    """Return a copy of ``df`` annotated with tagging-hygiene columns.

    Adds:

    * ``missing_tags`` -- list of required tags absent on that row,
    * ``n_missing_tags`` -- length of the above,
    * ``is_fully_tagged`` -- ``True`` when no required tag is missing.
    """
    required = list(required_tags)
    out = df.copy()

    present = pd.DataFrame(
        {tag: _is_present(out[tag]) for tag in required},
        index=out.index,
    )
    missing_lookup = {
        tag: pd.Series(tag, index=out.index).where(~present[tag], other=pd.NA)
        for tag in required
    }
    missing_frame = pd.DataFrame(missing_lookup)
    out["missing_tags"] = [
        [tag for tag in required if isinstance(row[tag], str)]
        for _, row in missing_frame.iterrows()
    ]
    out["n_missing_tags"] = out["missing_tags"].str.len().astype("int64")
    out["is_fully_tagged"] = out["n_missing_tags"] == 0
    return out


def tag_coverage(df: pd.DataFrame, required_tags: Iterable[str] = REQUIRED_TAGS) -> dict:
    """Compute row- and cost-weighted tag coverage plus per-tag presence.

    Returns a plain ``dict`` of Python scalars (JSON-serializable).
    """
    required = list(required_tags)
    flagged = add_tag_flags(df, required)

    total_rows = int(len(flagged))
    tagged_rows = int(flagged["is_fully_tagged"].sum())

    total_cost = float(flagged[COST_COLUMN].sum())
    tagged_cost = float(flagged.loc[flagged["is_fully_tagged"], COST_COLUMN].sum())

    per_tag = {
        tag: round(float(_is_present(df[tag]).mean() * 100.0), 4) if total_rows else 0.0
        for tag in required
    }

    return {
        "required_tags": required,
        "total_rows": total_rows,
        "tagged_rows": tagged_rows,
        "untagged_rows": total_rows - tagged_rows,
        "row_coverage_pct": round(tagged_rows / total_rows * 100.0, 4) if total_rows else 0.0,
        "total_cost": round(total_cost, 6),
        "tagged_cost": round(tagged_cost, 6),
        "untagged_cost": round(total_cost - tagged_cost, 6),
        "cost_coverage_pct": round(tagged_cost / total_cost * 100.0, 4) if total_cost else 0.0,
        "per_tag_coverage_pct": per_tag,
    }


def worst_offenders(
    df: pd.DataFrame,
    by: Union[str, Sequence[str]] = "service",
    required_tags: Iterable[str] = REQUIRED_TAGS,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank the dimensions carrying the most *untagged* spend.

    Parameters
    ----------
    by:
        Column (or list of columns) to group untagged rows by -- e.g.
        ``"service"``, ``"account_id"``, or ``["account_id", "service"]``.
    top_n:
        Keep only the ``top_n`` costliest groups.

    Returns a DataFrame with ``untagged_cost`` and ``untagged_rows`` columns,
    sorted by cost descending.
    """
    group_cols = [by] if isinstance(by, str) else list(by)
    flagged = add_tag_flags(df, required_tags)
    untagged = flagged[~flagged["is_fully_tagged"]]

    if untagged.empty:
        return pd.DataFrame(columns=[*group_cols, "untagged_cost", "untagged_rows"])

    grouped = (
        untagged.groupby(group_cols, dropna=False)[COST_COLUMN]
        .agg(untagged_cost="sum", untagged_rows="count")
        .reset_index()
        .sort_values("untagged_cost", ascending=False, kind="stable")
        .head(top_n)
        .reset_index(drop=True)
    )
    grouped["untagged_cost"] = grouped["untagged_cost"].round(6)
    grouped["untagged_rows"] = grouped["untagged_rows"].astype("int64")
    return grouped
