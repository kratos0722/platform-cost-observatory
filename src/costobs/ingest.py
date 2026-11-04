"""Load and validate CUR-style billing exports into a typed DataFrame.

The rest of the pipeline assumes the normalized frame produced by
:func:`load_cur`: ``usage_date`` is a ``datetime64`` day, the two numeric
columns are floats, and the free-text tag columns are a nullable string dtype
where blank / whitespace-only values have been collapsed to ``pd.NA`` so that
"empty string" and "missing" are treated identically everywhere downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

PathLike = Union[str, Path]

#: Allocation tags every resource is expected to carry.
REQUIRED_TAGS: tuple[str, ...] = ("owner", "team", "environment")

#: Columns a CUR-style export must provide, in canonical order.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "usage_date",
    "account_id",
    "service",
    "resource_id",
    "owner",
    "team",
    "environment",
    "unblended_cost",
    "usage_amount",
)

#: Columns that hold free-text identifiers / tags (kept as nullable strings).
_STRING_COLUMNS: tuple[str, ...] = (
    "account_id",
    "service",
    "resource_id",
    "owner",
    "team",
    "environment",
)

#: Columns that must parse as floating-point numbers.
_NUMERIC_COLUMNS: tuple[str, ...] = ("unblended_cost", "usage_amount")


class SchemaError(ValueError):
    """Raised when an input file is missing one or more required columns."""


def validate_schema(df: pd.DataFrame, required_columns: Iterable[str] = EXPECTED_COLUMNS) -> None:
    """Raise :class:`SchemaError` if any required column is absent."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"input is missing required column(s): {', '.join(missing)}; "
            f"expected at least {', '.join(required_columns)}"
        )


def _clean_tag_series(series: pd.Series) -> pd.Series:
    """Return a nullable-string copy with blank / whitespace values as ``pd.NA``."""
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.len() == 0, pd.NA)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw frame into the canonical typed schema.

    This is split out from :func:`load_cur` so callers that already hold a
    DataFrame (e.g. tests) can obtain the same normalization without touching
    disk.
    """
    validate_schema(df)
    out = df.copy()

    out["usage_date"] = pd.to_datetime(out["usage_date"], errors="raise").dt.normalize()

    for col in _NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")

    for col in _STRING_COLUMNS:
        out[col] = _clean_tag_series(out[col])

    # Return columns in canonical order, preserving any extras at the end.
    ordered = list(EXPECTED_COLUMNS) + [c for c in out.columns if c not in EXPECTED_COLUMNS]
    return out[ordered].reset_index(drop=True)


def load_cur(path: PathLike) -> pd.DataFrame:
    """Read a CUR-style CSV from ``path`` and return the normalized frame.

    Parameters
    ----------
    path:
        Filesystem path to a CSV export with at least :data:`EXPECTED_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SchemaError
        If the file is missing a required column.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CUR file not found: {p}")

    # Read tag/id columns as strings up front so empty cells stay distinguishable
    # from the literal text "nan"; numerics are coerced in ``normalize``.
    raw = pd.read_csv(p, dtype={col: "string" for col in _STRING_COLUMNS})
    return normalize(raw)
