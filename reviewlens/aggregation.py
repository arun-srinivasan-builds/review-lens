"""Deterministic weekly metrics. Pure pandas: the LLM never calculates numbers."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from reviewlens.models import CATEGORIES, SENTIMENTS, URGENCIES, WeeklyMetrics


def _counts(series: pd.Series, allowed: tuple[str, ...]) -> dict[str, int]:
    observed = series.value_counts()
    return {key: int(observed.get(key, 0)) for key in allowed}


def _period(enriched_df: pd.DataFrame) -> tuple[dt.date, dt.date]:
    today = dt.date.today()
    if enriched_df.empty or "date" not in enriched_df.columns:
        return today, today
    parsed = pd.to_datetime(enriched_df["date"], errors="coerce").dropna()
    if parsed.empty:
        return today, today
    return parsed.min().date(), parsed.max().date()


def calculate_weekly_metrics(enriched_df: pd.DataFrame) -> WeeklyMetrics:
    """Counts and percentages over completed rows; period spans the analysed rows (all rows if none)."""
    if "processing_status" in enriched_df.columns:
        completed_mask = enriched_df["processing_status"] == "completed"
    else:
        completed_mask = pd.Series(False, index=enriched_df.index)
    completed = enriched_df[completed_mask]

    total = len(completed)
    needs_review = int(len(enriched_df) - total)

    sentiment_counts = _counts(completed["sentiment"], SENTIMENTS) if total else dict.fromkeys(SENTIMENTS, 0)
    category_counts = _counts(completed["category"], CATEGORIES) if total else dict.fromkeys(CATEGORIES, 0)
    urgency_counts = _counts(completed["urgency"], URGENCIES) if total else dict.fromkeys(URGENCIES, 0)

    percentages = {
        key: round(count / total * 100, 1) if total else 0.0 for key, count in sentiment_counts.items()
    }
    period_start, period_end = _period(completed if total else enriched_df)

    return WeeklyMetrics(
        period_start=period_start,
        period_end=period_end,
        total_reviews=total,
        needs_review_count=needs_review,
        sentiment_counts=sentiment_counts,
        sentiment_percentages=percentages,
        category_counts=category_counts,
        urgency_counts=urgency_counts,
        high_urgency_count=urgency_counts["high"],
    )
