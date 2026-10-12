"""Calculated complaint breakdown: pandas only, platform concentration, no invented time patterns."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from reviewlens.aggregation import calculate_complaint_breakdown
from reviewlens.demo import load_demo_bundle
from reviewlens.models import ENRICHED_COLUMNS
from reviewlens.report import format_report_text
from tests.test_demo import _report

ROOT = Path(__file__).resolve().parent.parent


def _row(rid: str, platform: str, sentiment: str | None, category: str | None, urgency: str | None,
         status: str = "completed") -> dict[str, object]:
    return {
        "review_id": rid, "date": dt.date(2026, 10, 5), "platform": platform, "rating": 2.0,
        "review_text": "x", "sentiment": sentiment, "category": category, "urgency": urgency,
        "language": "English" if sentiment else None, "summary": "s" if sentiment else None,
        "suggested_reply": "r" if sentiment else None, "processing_status": status,
    }


def _df(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(ENRICHED_COLUMNS))


def test_counts_shares_platform_and_urgency() -> None:
    df = _df(
        [
            _row("1", "Swiggy", "negative", "delivery", "high"),
            _row("2", "Swiggy", "negative", "delivery", "medium"),
            _row("3", "Google", "negative", "delivery", "low"),
            _row("4", "Zomato", "negative", "price", "low"),
            _row("5", "Google", "positive", "delivery", "low"),  # not negative: ignored
            _row("6", "Swiggy", None, None, None, status="needs_review"),  # excluded
        ]
    )
    rows = calculate_complaint_breakdown(df)
    assert [r.category for r in rows] == ["delivery", "price"]
    delivery, price = rows
    assert (delivery.negative_reviews, delivery.share_of_negative_pct) == (3, 75.0)
    assert (delivery.top_platform, delivery.top_platform_reviews) == ("Swiggy", 2)
    assert delivery.high_urgency_reviews == 1
    assert (price.negative_reviews, price.share_of_negative_pct, price.top_platform) == (1, 25.0, "Zomato")


def test_ties_are_broken_deterministically() -> None:
    df = _df(
        [
            _row("1", "Zomato", "negative", "app", "low"),
            _row("2", "Google", "negative", "app", "low"),
            _row("3", "Swiggy", "negative", "staff", "low"),
            _row("4", "Swiggy", "negative", "staff", "low"),
            _row("5", "Zomato", "negative", "staff", "low"),
        ]
    )
    rows = calculate_complaint_breakdown(df)
    assert [r.category for r in rows] == ["staff", "app"]  # count first
    assert rows[1].top_platform == "Google"  # tie on count: alphabetical


def test_empty_and_no_negative_reviews() -> None:
    assert calculate_complaint_breakdown(_df([])) == []
    assert calculate_complaint_breakdown(_df([_row("1", "Google", "positive", "app", "low")])) == []


def test_top_n_limits_rows() -> None:
    df = _df([_row(str(i), "Google", "negative", c, "low") for i, c in enumerate(["app", "price", "staff", "app"])])
    assert len(calculate_complaint_breakdown(df, top_n=2)) == 2


def test_demo_bundle_breakdown_matches_independent_recomputation() -> None:
    df, _ = load_demo_bundle(ROOT / "demo_data")
    rows = calculate_complaint_breakdown(df)

    raw = pd.read_csv(ROOT / "demo_data" / "enriched.csv", dtype=str, keep_default_na=False)
    neg = raw[(raw.processing_status == "completed") & (raw.sentiment == "negative")]
    total = len(neg)
    assert sum(r.negative_reviews for r in rows) == total
    assert round(sum(r.share_of_negative_pct for r in rows), 0) == 100
    for r in rows:
        cat = neg[neg.category == r.category]
        assert r.negative_reviews == len(cat)
        counts = cat.platform.value_counts()
        assert r.top_platform_reviews == int(counts.max())
        assert counts[r.top_platform] == counts.max()
        assert r.high_urgency_reviews == int((cat.urgency == "high").sum())
        assert r.share_of_negative_pct == round(len(cat) / total * 100, 1)


def test_text_report_includes_calculated_block_only_when_given() -> None:
    df, report = load_demo_bundle(ROOT / "demo_data")
    from reviewlens.aggregation import calculate_weekly_metrics

    metrics = calculate_weekly_metrics(df)
    plain = format_report_text(metrics, report)
    assert "COMPLAINT BREAKDOWN" not in plain
    full = format_report_text(metrics, report, calculate_complaint_breakdown(df, top_n=3))
    assert "COMPLAINT BREAKDOWN (calculated from analysed reviews)" in full
    first = calculate_complaint_breakdown(df)[0]
    assert f"{first.negative_reviews} negative reviews" in full


def test_no_time_of_day_wording_is_ever_generated() -> None:
    df, report = load_demo_bundle(ROOT / "demo_data")
    from reviewlens.aggregation import calculate_weekly_metrics

    text = format_report_text(
        calculate_weekly_metrics(df), report, calculate_complaint_breakdown(df, top_n=3)
    )
    block = text.split("COMPLAINT BREAKDOWN", 1)[1].split("TOP PRAISE", 1)[0].lower()
    for forbidden in ("pm", "am ", "evening", "morning", "night", "o'clock", "peak hour", "slot"):
        assert forbidden not in block
    assert report is not None and _report() is not None
