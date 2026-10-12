"""Compact period labels for KPI cards (no truncation in a quarter-width card)."""

from __future__ import annotations

import datetime as dt

import pytest

from ui.dashboard import format_period


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (dt.date(2026, 10, 5), dt.date(2026, 10, 11), "5–11 Oct 2026"),
        (dt.date(2026, 9, 28), dt.date(2026, 10, 5), "28 Sep – 5 Oct 2026"),
        (dt.date(2025, 12, 29), dt.date(2026, 1, 4), "29 Dec 2025 – 4 Jan 2026"),
        (dt.date(2026, 10, 5), dt.date(2026, 10, 5), "5 Oct 2026"),
    ],
)
def test_format_period(start: dt.date, end: dt.date, expected: str) -> None:
    assert format_period(start, end) == expected


def test_period_label_is_short_enough_for_a_kpi_card() -> None:
    longest = format_period(dt.date(2025, 12, 29), dt.date(2026, 1, 4))
    typical = format_period(dt.date(2026, 10, 5), dt.date(2026, 10, 11))
    assert len(typical) <= 14  # fits at the KPI font size without an ellipsis
    assert len(longest) <= 30


def test_weekly_report_kpis_are_short_and_have_no_trend_arrow() -> None:
    import inspect

    from ui import dashboard

    source = inspect.getsource(dashboard._report_tab)
    assert 'delta_arrow="off"' in source  # a sentiment split must not look like a trend
    assert "high-urgency review(s)" not in source  # long value that used to be truncated
