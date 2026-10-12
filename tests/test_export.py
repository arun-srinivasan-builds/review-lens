"""Tests for reviewlens.export (formula neutralisation and CSV bytes)."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

from reviewlens.export import export_enriched_csv, neutralize_formula
from reviewlens.models import ENRICHED_COLUMNS


def _df(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "review_id": "R-1",
        "date": "2026-10-03",
        "platform": "Zomato",
        "rating": 4.0,
        "review_text": "Nalla irundhuchu",
        "sentiment": "positive",
        "category": "food_quality",
        "urgency": "low",
        "language": "Tanglish",
        "summary": "Good food.",
        "suggested_reply": "Thank you!",
        "processing_status": "completed",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _parse(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", keep_default_na=False, dtype=str)


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_dangerous_prefixes_neutralised(prefix: str) -> None:
    assert neutralize_formula(f"{prefix}SUM(A1)") == f"'{prefix}SUM(A1)"


@pytest.mark.parametrize("value", [" =1+1", "  @cmd", "\t=1", "\r+1", "   -2"])
def test_leading_whitespace_variants(value: str) -> None:
    assert neutralize_formula(value) == "'" + value


@pytest.mark.parametrize("value", ["Great food", "", "a=b", "5 stars - nice", "Nalla irundhuchu"])
def test_normal_text_unchanged(value: str) -> None:
    assert neutralize_formula(value) == value


def test_legitimate_dash_text_is_still_neutralised() -> None:
    # Documented trade-off: safety over fidelity for free text beginning with '-'.
    assert neutralize_formula("-") == "'-"
    assert neutralize_formula("- bad service") == "'- bad service"


def test_bom_and_unicode_round_trip() -> None:
    text = "உணவு நன்றாக இருந்தது"
    data = export_enriched_csv(_df(review_text=text))
    assert data.startswith(b"\xef\xbb\xbf")
    assert _parse(data).loc[0, "review_text"] == text


def test_rating_numeric_and_unmodified() -> None:
    out = _parse(export_enriched_csv(_df(rating=4.5)))
    assert float(out.loc[0, "rating"]) == 4.5
    assert not out.loc[0, "rating"].startswith("'")


def test_column_order_and_extra_columns_dropped() -> None:
    df = _df()
    df["extra"] = "x"
    df = df[["extra", *reversed(ENRICHED_COLUMNS)]]
    out = _parse(export_enriched_csv(df))
    assert list(out.columns) == list(ENRICHED_COLUMNS)


def test_all_text_columns_sanitised() -> None:
    df = _df(
        review_id="=A1",
        platform="+x",
        review_text='=HYPERLINK("http://evil")',
        summary="@sum",
        suggested_reply="-cmd",
        language=" =x",
    )
    out = _parse(export_enriched_csv(df)).iloc[0]
    for col in ("review_id", "platform", "review_text", "summary", "suggested_reply", "language"):
        assert out[col].startswith("'"), col


def test_none_and_nan_become_empty() -> None:
    df = _df(summary=None, suggested_reply=np.nan, sentiment=None, language=pd.NA)
    out = _parse(export_enriched_csv(df)).iloc[0]
    for col in ("summary", "suggested_reply", "sentiment", "language"):
        assert out[col] == ""


def test_dates_are_iso() -> None:
    out = _parse(export_enriched_csv(_df(date=dt.date(2026, 10, 3)))).iloc[0]
    assert out["date"] == "2026-10-03"
