"""Academy compliance: invalid / failing LLM responses are retried a bounded number of times.

All LLM behaviour is mocked. The fake model feeds raw dicts through the real `ReviewAnalysis` schema, as
`ChatOpenAI.with_structured_output(ReviewAnalysis)` does, so invalid enum values raise a genuine pydantic
ValidationError. No network or API key is involved.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from typing import Any

import pandas as pd
import pytest
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import SecretStr

from reviewlens import classification
from reviewlens.aggregation import calculate_weekly_metrics
from reviewlens.classification import classify_reviews, enriched_to_dataframe
from reviewlens.config import Settings
from reviewlens.models import REQUIRED_COLUMNS, ReviewAnalysis

SECRET = "sk-test-RETRYSUITE0123456789"
MAX_RETRIES = 2

VALID: dict[str, str] = {
    "sentiment": "negative",
    "category": "delivery",
    "urgency": "high",
    "language": "English",
    "summary": "The order arrived late and cold.",
    "suggested_reply": "We are sorry. Please contact our support team.",
}


def _settings(max_retries: int = MAX_RETRIES) -> Settings:
    return Settings(
        openai_api_key=SecretStr(SECRET),
        openai_model="fake-model",
        max_retries=max_retries,
        max_concurrency=2,
    )


class TemporaryError(Exception):
    """Named like OpenAI's RateLimitError so the retry policy classifies it as transient."""


TemporaryError.__name__ = "RateLimitError"


class ScriptedLLM:
    """Replays a per-review script. Each item is a raw dict (validated by the real schema) or an exception.

    When a script is exhausted its last item repeats, so "always invalid" is a one-item script.
    `attempts[text]` records how many times the model was invoked for that review.
    """

    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = scripts
        self.attempts: dict[str, int] = defaultdict(int)

    def with_structured_output(self, schema: Any) -> Runnable:
        assert schema is ReviewAnalysis

        def run(prompt_value: Any) -> Any:
            body = str(prompt_value.to_messages()[-1].content)
            text = body.removeprefix("<review>\n").removesuffix("\n</review>")
            script = self.scripts.get(text, [VALID])
            item = script[min(self.attempts[text], len(script) - 1)]
            self.attempts[text] += 1
            if isinstance(item, BaseException):
                raise item
            return schema.model_validate(item)  # raises pydantic.ValidationError on a bad enum

        return RunnableLambda(run)


def _reviews(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"R{i}" for i in range(len(texts))],
            "date": [dt.date(2026, 10, 5)] * len(texts),
            "platform": ["Google"] * len(texts),
            "rating": [2.0] * len(texts),
            "review_text": texts,
        },
        columns=list(REQUIRED_COLUMNS),
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(classification, "_sleep", sleeps.append)
    return sleeps


# ---- scenarios 1-3: an invalid enum on the first response, valid on the retry
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        pytest.param("category", "transport", id="1-invalid-category-transport"),
        pytest.param("sentiment", "angry", id="2-invalid-sentiment-angry"),
        pytest.param("urgency", "critical", id="3-invalid-urgency-critical"),
    ],
)
def test_invalid_enum_then_valid_is_retried_and_completed(
    field: str, bad_value: str, _no_sleep: list[float]
) -> None:
    text = "food was cold and late"
    llm = ScriptedLLM({text: [{**VALID, field: bad_value}, VALID]})

    [result] = classify_reviews(_reviews([text]), _settings(), llm=llm)  # type: ignore[arg-type]

    assert llm.attempts[text] == 2  # first invalid + one retry
    assert result.processing_status == "completed"
    assert getattr(result, field) == VALID[field]  # the valid value, never the rejected one
    assert result.summary == VALID["summary"]
    assert len(_no_sleep) == 1  # exactly one backoff before the retry


# ---- scenario 4: persistent invalid data stops at the configured limit
@pytest.mark.parametrize("max_retries", [0, 1, 2, 4])
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("category", "transport"), ("sentiment", "angry"), ("urgency", "critical")],
)
def test_persistent_invalid_data_stops_at_configured_limit(
    field: str, bad_value: str, max_retries: int, _no_sleep: list[float]
) -> None:
    text = "always invalid response"
    llm = ScriptedLLM({text: [{**VALID, field: bad_value}]})

    [result] = classify_reviews(_reviews([text]), _settings(max_retries), llm=llm)  # type: ignore[arg-type]

    assert llm.attempts[text] == 1 + max_retries  # bounded, not indefinite
    assert result.processing_status == "needs_review"
    assert result.review_text == text  # source preserved
    assert result.sentiment is None and result.category is None and result.urgency is None
    assert result.summary is None and result.suggested_reply is None  # nothing fabricated
    assert len(_no_sleep) == max_retries


# ---- scenario 5: temporary provider errors follow the policy and never loop forever
def test_temporary_error_then_success_is_retried(_no_sleep: list[float]) -> None:
    text = "provider hiccup"
    llm = ScriptedLLM({text: [TemporaryError("429 slow down"), VALID]})

    [result] = classify_reviews(_reviews([text]), _settings(), llm=llm)  # type: ignore[arg-type]

    assert llm.attempts[text] == 2
    assert result.processing_status == "completed"
    assert len(_no_sleep) == 1


@pytest.mark.parametrize("max_retries", [0, 1, 2, 3])
def test_persistent_temporary_error_is_bounded(max_retries: int, _no_sleep: list[float]) -> None:
    text = "provider down"
    llm = ScriptedLLM({text: [TemporaryError("503 unavailable")]})

    [result] = classify_reviews(_reviews([text]), _settings(max_retries), llm=llm)  # type: ignore[arg-type]

    assert llm.attempts[text] == 1 + max_retries
    assert result.processing_status == "needs_review"
    assert result.review_text == text
    assert len(_no_sleep) == max_retries
    assert all(s <= 8.0 for s in _no_sleep)  # backoff is capped


def test_builtin_timeout_is_treated_as_temporary(_no_sleep: list[float]) -> None:
    text = "slow provider"
    llm = ScriptedLLM({text: [TimeoutError("timed out"), VALID]})

    [result] = classify_reviews(_reviews([text]), _settings(), llm=llm)  # type: ignore[arg-type]

    assert llm.attempts[text] == 2
    assert result.processing_status == "completed"


# ---- scenario 6: one failing review does not affect the rest, and aggregation excludes it
def test_one_failure_preserves_successes_reports_failure_and_is_excluded_from_metrics(
    caplog: pytest.LogCaptureFixture, _no_sleep: list[float]
) -> None:
    texts = ["good one", "unfixable review", "good two", "good three", "good four"]
    ok_positive = {**VALID, "sentiment": "positive", "category": "food_quality", "urgency": "low"}
    llm = ScriptedLLM(
        {
            "unfixable review": [{**VALID, "urgency": "critical"}],
            "good two": [ok_positive],
            "good four": [ok_positive],
        }
    )

    with caplog.at_level(logging.INFO):
        results = classify_reviews(_reviews(texts), _settings(), llm=llm)  # type: ignore[arg-type]

    # order preserved; only the bad review is flagged
    assert [r.review_id for r in results] == ["R0", "R1", "R2", "R3", "R4"]
    assert [r.processing_status for r in results] == [
        "completed",
        "needs_review",
        "completed",
        "completed",
        "completed",
    ]
    # successful reviews keep their full analysis; no duplicate calls for them
    assert all(r.summary and r.suggested_reply for r in results if r.processing_status == "completed")
    assert all(llm.attempts[t] == 1 for t in texts if t != "unfixable review")
    assert llm.attempts["unfixable review"] == 1 + MAX_RETRIES

    # the failure is reported (log names the class only, never review text or the key)
    log_text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "needs_review" in log_text
    assert "unfixable review" not in log_text
    assert SECRET not in log_text

    # aggregation: totals count only successful reviews; the failure is reported separately
    metrics = calculate_weekly_metrics(enriched_to_dataframe(results))
    assert metrics.total_reviews == 4
    assert metrics.needs_review_count == 1
    assert sum(metrics.sentiment_counts.values()) == 4
    assert sum(metrics.category_counts.values()) == 4
    assert sum(metrics.urgency_counts.values()) == 4
    assert metrics.sentiment_counts == {"positive": 2, "neutral": 0, "negative": 2}
    assert metrics.high_urgency_count == 2  # the failed review's rejected "critical" never counted
