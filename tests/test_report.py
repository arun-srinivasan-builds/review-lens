from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest
from pydantic import SecretStr

from reviewlens import report as report_mod
from reviewlens.config import Settings
from reviewlens.models import ENRICHED_COLUMNS, WeeklyMetrics, WeeklyReport
from reviewlens.report import (
    ReportGenerationError,
    build_review_evidence,
    format_report_text,
    generate_weekly_report,
)

SECRET = "sk-test1234567890abcdef"


def _settings(retries: int = 2) -> Settings:
    return Settings(openai_api_key=SecretStr(SECRET), openai_model="m", max_retries=retries)


def _metrics(total: int = 10) -> WeeklyMetrics:
    return WeeklyMetrics(
        period_start=dt.date(2026, 1, 1),
        period_end=dt.date(2026, 1, 7),
        total_reviews=total,
        needs_review_count=0,
        sentiment_counts={"positive": 5, "neutral": 2, "negative": 3},
        sentiment_percentages={"positive": 50.0, "neutral": 20.0, "negative": 30.0},
        category_counts={"delivery": 3},
        urgency_counts={"low": 6, "medium": 1, "high": 3},
        high_urgency_count=3,
    )


def _good_report() -> WeeklyReport:
    return WeeklyReport(
        top_3_complaints=["Late delivery"],
        top_praises=["Tasty food"],
        urgent_attention="Cold food complaints need a response.",
        action_items=["a", "b", "c"],
    )


class FakeChain:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.payloads: list[dict[str, str]] = []

    def invoke(self, payload: dict[str, str]) -> Any:
        self.calls += 1
        self.payloads.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class AuthenticationError(Exception):
    pass


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(report_mod, "_sleep", sleeps.append)
    return sleeps


def _patch_chain(monkeypatch: pytest.MonkeyPatch, chain: FakeChain) -> None:
    monkeypatch.setattr(report_mod, "build_report_chain", lambda llm=None, settings=None: chain)


def test_success_passes_metrics_and_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = FakeChain([_good_report()])
    _patch_chain(monkeypatch, chain)
    result = generate_weekly_report(_metrics(), ["[1 | Zomato | negative | delivery | high] Late"], settings=_settings())
    assert result.action_items == ["a", "b", "c"]
    assert '"total_reviews": 10' in chain.payloads[0]["metrics_json"]
    assert "Late" in chain.payloads[0]["evidence"]


def test_exactly_three_action_items_enforced() -> None:
    with pytest.raises(ValueError):
        WeeklyReport(top_3_complaints=[], top_praises=[], urgent_attention="x", action_items=["a", "b"])
    with pytest.raises(ValueError):
        WeeklyReport(top_3_complaints=[], top_praises=[], urgent_attention="x", action_items=["a", "b", "c", "d"])


def test_retry_then_success(monkeypatch: pytest.MonkeyPatch, _no_sleep: list[float]) -> None:
    chain = FakeChain([ValueError("bad output"), _good_report()])
    _patch_chain(monkeypatch, chain)
    generate_weekly_report(_metrics(), [], settings=_settings(retries=2))
    assert chain.calls == 2
    assert len(_no_sleep) == 1


def test_retry_exhausted_raises_secret_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = FakeChain([RuntimeError(f"boom {SECRET}")] * 3)
    _patch_chain(monkeypatch, chain)
    with pytest.raises(ReportGenerationError) as info:
        generate_weekly_report(_metrics(), [], settings=_settings(retries=2))
    assert chain.calls == 3
    assert SECRET not in str(info.value)


def test_auth_error_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    chain = FakeChain([AuthenticationError(f"bad key {SECRET}"), _good_report()])
    _patch_chain(monkeypatch, chain)
    with pytest.raises(ReportGenerationError) as info:
        generate_weekly_report(_metrics(), [], settings=_settings(retries=3))
    assert chain.calls == 1
    assert SECRET not in str(info.value)


def test_zero_reviews_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("LLM chain must not be built")

    monkeypatch.setattr(report_mod, "build_report_chain", boom)
    with pytest.raises(ReportGenerationError, match="No successfully analysed reviews"):
        generate_weekly_report(_metrics(total=0), [], settings=_settings())


def _evidence_df() -> pd.DataFrame:
    base = {"date": dt.date(2026, 1, 1), "rating": 3, "review_text": "RAW TEXT SECRET", "language": "English",
            "suggested_reply": "r"}
    rows = [
        {**base, "review_id": "c", "platform": "Swiggy", "sentiment": "positive", "category": "app",
         "urgency": "low", "summary": "Nice app", "processing_status": "completed"},
        {**base, "review_id": "b", "platform": "Zomato", "sentiment": "negative", "category": "price",
         "urgency": "medium", "summary": "Too pricey", "processing_status": "completed"},
        {**base, "review_id": "a", "platform": "Zomato", "sentiment": "negative", "category": "delivery",
         "urgency": "high", "summary": "Food arrived cold", "processing_status": "completed"},
        {**base, "review_id": "d", "platform": "Zomato", "sentiment": None, "category": None,
         "urgency": None, "summary": None, "processing_status": "needs_review"},
    ]
    return pd.DataFrame(rows, columns=list(ENRICHED_COLUMNS))


def test_evidence_from_summaries_ordered_and_capped() -> None:
    lines = build_review_evidence(_evidence_df())
    assert lines == [
        "[a | Zomato | negative | delivery | high] Food arrived cold",
        "[b | Zomato | negative | price | medium] Too pricey",
        "[c | Swiggy | positive | app | low] Nice app",
    ]
    assert not any("RAW TEXT" in line for line in lines)
    assert build_review_evidence(_evidence_df(), max_items=2) == lines[:2]
    assert build_review_evidence(_evidence_df()) == lines  # deterministic


def test_format_report_text_uses_metrics_numbers() -> None:
    rep = _good_report()
    rep = rep.model_copy(update={"urgent_attention": "99 reviews are urgent!"})
    text = format_report_text(_metrics(), rep)
    assert "WEEKLY FEEDBACK REPORT | 2026-01-01 to 2026-01-07" in text
    assert "Total analysed reviews: 10" in text
    assert "Positive: 50.0% | Neutral: 20.0% | Negative: 30.0%" in text
    assert "3 reviews require attention" in text
    assert "1. a" in text and "3. c" in text
