"""CSV export with spreadsheet formula-injection protection."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from reviewlens.models import ENRICHED_COLUMNS

_DANGEROUS_PREFIXES = ("=", "+", "-", "@")
_LEADING_CONTROL = ("\t", "\r", "\n")
_NUMERIC_COLUMNS = ("rating",)


def neutralize_formula(value: str) -> str:
    """Prefix a text cell with ``'`` if a spreadsheet could treat it as a formula.

    A cell is dangerous when, after any leading whitespace, it starts with
    ``=``, ``+``, ``-`` or ``@``, or when it starts with a tab/CR (which Excel
    can also interpret). Note this deliberately also neutralises harmless free
    text such as "-" or "-5 minutes late"; safety wins over fidelity.
    """
    if not isinstance(value, str) or value == "":
        return value
    if value.startswith(_LEADING_CONTROL):
        return "'" + value
    if value.lstrip().startswith(_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _text_cell(value: object) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
        return value.date().isoformat() if isinstance(value, (dt.datetime, pd.Timestamp)) else value.isoformat()
    return neutralize_formula(str(value))


def export_enriched_csv(enriched_df: pd.DataFrame) -> bytes:
    """Serialise the enriched table as UTF-8-with-BOM CSV, safe to open in Excel."""
    out = pd.DataFrame(index=range(len(enriched_df)))
    for col in ENRICHED_COLUMNS:
        if col not in enriched_df.columns:
            out[col] = ""
            continue
        series = enriched_df[col].reset_index(drop=True)
        if col in _NUMERIC_COLUMNS:
            out[col] = pd.to_numeric(series, errors="coerce")
        else:
            out[col] = series.map(_text_cell)
    csv_text = out.to_csv(index=False, na_rep="")
    return csv_text.encode("utf-8-sig")
