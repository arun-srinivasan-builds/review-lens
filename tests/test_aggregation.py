from __future__ import annotations

import datetime as dt

import pandas as pd

from reviewlens.aggregation import calculate_weekly_metrics
from reviewlens.models import CATEGORIES, ENRICHED_COLUMNS, SENTIMENTS, URGENCIES


def _row(rid: str, date: dt.date, sentiment, category, urgency, status="completed") -> dict:
    return {
        "review_id": rid,
        "date": date,
        "platform": "Zomato",
        "rating": 3,
        "review_text": "text",
        "sentiment": sentiment,
        "category": category,
        "urgency": urgency,
        "language": "English" if sentiment else None,
        "summary": "s" if sentiment else None,
        "suggested_reply": "r" if sentiment else None,
        "processing_status": status,
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(ENRICHED_COLUMNS))


def test_exact_counts_and_percentages() -> None:
    d = dt.date
    df = _df(
        [
            _row("1", d(2026, 1, 5), "negative", "delivery", "high"),
            _row("2", d(2026, 1, 7), "negative", "delivery", "medium"),
            _row("3", d(2026, 1, 9), "positive", "food_quality", "low"),
            _row("4", d(2026, 1, 3), "neutral", "price", "low"),
            _row("5", d(2026, 1, 11), None, None, None, status="needs_review"),
        ]
    )
    m = calculate_weekly_metrics(df)
    assert m.total_reviews == 4
    assert m.needs_review_count == 1
    assert m.sentiment_counts == {"positive": 1, "neutral": 1, "negative": 2}
    assert m.sentiment_percentages == {"positive": 25.0, "neutral": 25.0, "negative": 50.0}
    assert m.category_counts["delivery"] == 2
    assert m.category_counts["food_quality"] == 1
    assert m.urgency_counts == {"low": 2, "medium": 1, "high": 1}
    assert m.high_urgency_count == 1
    assert m.period_start == d(2026, 1, 3)
    assert m.period_end == d(2026, 1, 9)  # period spans analysed rows only


def test_all_keys_present_and_rounding() -> None:
    df = _df([_row(str(i), dt.date(2026, 1, 1), "positive" if i == 0 else "negative", "app", "low") for i in range(3)])
    m = calculate_weekly_metrics(df)
    assert set(m.sentiment_counts) == set(SENTIMENTS)
    assert set(m.category_counts) == set(CATEGORIES)
    assert set(m.urgency_counts) == set(URGENCIES)
    assert m.category_counts["staff"] == 0
    assert m.sentiment_percentages["positive"] == 33.3
    assert m.sentiment_percentages["negative"] == 66.7


def test_zero_completed() -> None:
    df = _df(
        [
            _row("1", dt.date(2026, 2, 1), None, None, None, status="needs_review"),
            _row("2", dt.date(2026, 2, 8), None, None, None, status="needs_review"),
        ]
    )
    m = calculate_weekly_metrics(df)
    assert m.total_reviews == 0
    assert m.needs_review_count == 2
    assert all(v == 0.0 for v in m.sentiment_percentages.values())
    assert m.high_urgency_count == 0
    assert (m.period_start, m.period_end) == (dt.date(2026, 2, 1), dt.date(2026, 2, 8))


def test_empty_frame_uses_today() -> None:
    m = calculate_weekly_metrics(_df([]))
    assert m.total_reviews == 0
    assert m.period_start == m.period_end == dt.date.today()


def test_string_dates_parsed() -> None:
    df = _df([_row("1", "2026-03-02", "positive", "app", "low"), _row("2", "2026-03-09", "neutral", "app", "low")])
    m = calculate_weekly_metrics(df)
    assert (m.period_start, m.period_end) == (dt.date(2026, 3, 2), dt.date(2026, 3, 9))
