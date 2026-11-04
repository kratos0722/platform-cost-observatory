"""Platform Cost Observatory.

A deterministic, dependency-light analytics pipeline over AWS Cost & Usage
Report (CUR)-style billing exports. It turns raw line items into four
governance signals:

1. Tagging hygiene  -- who is missing required allocation tags, and how much
   spend is unattributable as a result.
2. Cost allocation  -- clean cost rollups per owner / team / service / day,
   with an explicit ``unallocated`` bucket for untagged spend.
3. Forecasting      -- month-end spend projections per owner from the
   month-to-date run rate (deterministic and explainable).
4. Anomaly detection -- per-entity spend spikes flagged against a rolling
   mean +/- k*stddev baseline, each with a human-readable reason.

The library is intentionally free of network access and randomness so that
results are reproducible in tests and CI.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import allocation, anomaly, forecast, ingest, report, tagging
from .ingest import REQUIRED_TAGS, load_cur

__all__ = [
    "__version__",
    "REQUIRED_TAGS",
    "load_cur",
    "ingest",
    "tagging",
    "allocation",
    "forecast",
    "anomaly",
    "report",
]
