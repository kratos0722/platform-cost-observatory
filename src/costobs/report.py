"""Assemble the four signals into one JSON-serializable report + a text view."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import pandas as pd

from . import allocation, anomaly, budget, forecast, tagging
from .forecast import TimestampLike
from .ingest import REQUIRED_TAGS


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts (Timestamps -> ISO date strings)."""
    if df.empty:
        return []
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
        elif safe[col].dtype == "object":
            safe[col] = safe[col].map(lambda v: list(v) if isinstance(v, list) else v)
    safe = safe.convert_dtypes()
    return [
        {k: (None if pd.isna(v) else v) for k, v in rec.items()}
        for rec in safe.astype(object).where(pd.notna(safe), None).to_dict("records")
    ]


def build_report(
    df: pd.DataFrame,
    required_tags: Iterable[str] = REQUIRED_TAGS,
    as_of: Optional[TimestampLike] = None,
    k: float = 3.0,
    window: int = 7,
    top_n: int = 10,
    forecast_method: str = "run_rate",
    anomaly_group: Iterable[str] = ("owner",),
    min_rel_increase: float = 0.25,
    budgets: Optional[Mapping[str, float]] = None,
) -> dict:
    """Run the full pipeline and return a nested, JSON-serializable dict."""
    required = list(required_tags)

    coverage = tagging.tag_coverage(df, required)
    offenders = tagging.worst_offenders(df, by="service", required_tags=required, top_n=top_n)

    alloc_owner = allocation.by_owner(df).head(top_n)
    alloc_team = allocation.by_team(df).head(top_n)
    alloc_service = allocation.by_service(df).head(top_n)

    fc = forecast.forecast_month_end(df, as_of=as_of, method=forecast_method)
    budget_rows = budget.check_budgets(fc, dict(budgets)) if budgets else pd.DataFrame()
    anoms = anomaly.detect_anomalies(
        df,
        group_cols=list(anomaly_group),
        k=k,
        window=window,
        min_rel_increase=min_rel_increase,
    )

    dates = pd.to_datetime(df["usage_date"])
    return {
        "meta": {
            "rows": int(len(df)),
            "date_range": {
                "start": dates.min().strftime("%Y-%m-%d"),
                "end": dates.max().strftime("%Y-%m-%d"),
            },
            "accounts": sorted(df["account_id"].dropna().unique().tolist()),
            "total_cost": round(float(df["unblended_cost"].sum()), 6),
            "params": {
                "required_tags": required,
                "forecast_method": forecast_method,
                "anomaly_group": list(anomaly_group),
                "k": k,
                "window": window,
                "min_rel_increase": min_rel_increase,
                "top_n": top_n,
            },
        },
        "tagging": {
            "coverage": coverage,
            "worst_offenders_by_service": _records(offenders),
        },
        "allocation": {
            "by_owner": _records(alloc_owner),
            "by_team": _records(alloc_team),
            "by_service": _records(alloc_service),
        },
        "forecast": _records(fc),
        "budgets": _records(budget_rows),
        "anomalies": _records(anoms),
    }


def _money(value: float) -> str:
    return f"${value:,.2f}"


def render_text(report: dict) -> str:
    """Render :func:`build_report` output as a readable multi-section string."""
    meta = report["meta"]
    cov = report["tagging"]["coverage"]
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("PLATFORM COST OBSERVATORY -- cost governance report")
    lines.append("=" * 70)
    lines.append(
        f"rows={meta['rows']:,}  "
        f"range={meta['date_range']['start']}..{meta['date_range']['end']}  "
        f"accounts={len(meta['accounts'])}  "
        f"total={_money(meta['total_cost'])}"
    )

    lines.append("")
    lines.append("-- Tagging hygiene " + "-" * 51)
    lines.append(
        f"row coverage : {cov['row_coverage_pct']:.1f}%  "
        f"({cov['tagged_rows']:,}/{cov['total_rows']:,} rows fully tagged)"
    )
    lines.append(
        f"cost coverage: {cov['cost_coverage_pct']:.1f}%  "
        f"(unallocated {_money(cov['untagged_cost'])} of {_money(cov['total_cost'])})"
    )
    per_tag = "  ".join(f"{t}={p:.1f}%" for t, p in cov["per_tag_coverage_pct"].items())
    lines.append(f"per-tag      : {per_tag}")
    offenders = report["tagging"]["worst_offenders_by_service"]
    if offenders:
        lines.append("worst offenders by service (untagged spend):")
        for row in offenders[:5]:
            lines.append(
                f"    {row['service']:<22} {_money(row['untagged_cost']):>14}  "
                f"({row['untagged_rows']} rows)"
            )

    lines.append("")
    lines.append("-- Cost allocation (top owners) " + "-" * 38)
    for row in report["allocation"]["by_owner"][:10]:
        lines.append(f"    {str(row['owner']):<22} {_money(row['cost']):>14}")

    lines.append("")
    lines.append("-- Month-end forecast " + "-" * 48)
    fc = report["forecast"]
    if fc:
        month = fc[0]["month"]
        as_of = fc[0]["as_of"]
        lines.append(f"month={month}  as_of={as_of}  method={fc[0]['method']}")
        for row in fc[:10]:
            lines.append(
                f"    {str(row['owner']):<22} MTD {_money(row['mtd_cost']):>12}  "
                f"-> projected {_money(row['projected_month_end']):>12}"
            )
    else:
        lines.append("    (no spend in the forecast month)")

    budgets = report.get("budgets") or []
    if budgets:
        lines.append("")
        lines.append("-- Budget tracking " + "-" * 51)
        for row in budgets:
            if row["status"] == "no_budget":
                continue
            flag = "[OVER]" if row["status"] == "over" else "  ok  "
            lines.append(
                f"    {flag} {str(row['owner']):<18} "
                f"proj {_money(row['projected_month_end']):>12} / "
                f"budget {_money(row['budget']):>12}  ({row['pct_of_budget']:.0f}%)"
            )

    lines.append("")
    lines.append("-- Anomalies " + "-" * 57)
    anoms = report["anomalies"]
    if anoms:
        for row in anoms:
            lines.append(f"    [!] {row['reason']}")
    else:
        lines.append("    none detected")

    lines.append("=" * 70)
    return "\n".join(lines)
