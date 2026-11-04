"""Command-line entry point: ``python -m costobs`` / ``costobs``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .ingest import REQUIRED_TAGS, load_cur
from .report import build_report, render_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="costobs",
        description=(
            "Platform Cost Observatory -- analyze a CUR-style billing export for "
            "tagging hygiene, cost allocation, month-end forecast and anomalies."
        ),
    )
    parser.add_argument("--version", action="version", version=f"costobs {__version__}")
    parser.add_argument(
        "-i", "--input", required=True, help="path to a CUR-style CSV export"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full report as JSON instead of the text summary",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="forecast 'today' (YYYY-MM-DD); defaults to the latest usage_date",
    )
    parser.add_argument(
        "--forecast-method",
        choices=("run_rate", "linear"),
        default="run_rate",
        help="forecasting method (default: run_rate)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help="anomaly threshold in standard deviations (default: 3.0)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="rolling baseline length in days for anomaly detection (default: 7)",
    )
    parser.add_argument(
        "--min-rel",
        type=float,
        default=0.25,
        help="minimum fraction above baseline mean to flag an anomaly (default: 0.25)",
    )
    parser.add_argument(
        "--anomaly-group",
        default="owner",
        help="comma-separated columns defining an anomaly entity (default: owner)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="number of rows to keep in ranked sections (default: 10)",
    )
    parser.add_argument(
        "--required-tags",
        default=",".join(REQUIRED_TAGS),
        help="comma-separated required allocation tags (default: owner,team,environment)",
    )
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    df = load_cur(args.input)
    report = build_report(
        df,
        required_tags=_split_csv(args.required_tags),
        as_of=args.as_of,
        k=args.k,
        window=args.window,
        top_n=args.top,
        forecast_method=args.forecast_method,
        anomaly_group=_split_csv(args.anomaly_group),
        min_rel_increase=args.min_rel,
    )

    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
