"""Pydantic and dataclass models shared across ReviewLens modules."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

import pandas as pd
from pydantic import AfterValidator, BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Category = Literal["food_quality", "delivery", "price", "staff", "packaging", "app"]
Urgency = Literal["low", "medium", "high"]
ProcessingStatus = Literal["completed", "needs_review"]

SENTIMENTS: tuple[str, ...] = ("positive", "neutral", "negative")
CATEGORIES: tuple[str, ...] = (
    "food_quality",
    "delivery",
    "price",
    "staff",
    "packaging",
    "app",
)
URGENCIES: tuple[str, ...] = ("low", "medium", "high")

REQUIRED_COLUMNS: tuple[str, ...] = ("review_id", "date", "platform", "rating", "review_text")

ENRICHED_COLUMNS: tuple[str, ...] = (
    "review_id",
    "date",
    "platform",
    "rating",
    "review_text",
    "sentiment",
    "category",
    "urgency",
    "language",
    "summary",
    "suggested_reply",
    "processing_status",
)


MAX_LANGUAGE_CHARS = 40
MAX_SUMMARY_CHARS = 300
MAX_REPLY_CHARS = 600


def _clean_text(max_chars: int) -> Callable[[str], str]:
    """Collapse whitespace, reject blanks and cap length (post-hoc, so it stays out of the strict JSON schema)."""

    def check(value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("must not be blank")
        return collapsed[:max_chars]

    return check


def _exactly(n: int) -> Callable[[list[str]], list[str]]:
    def check(value: list[str]) -> list[str]:
        if len(value) != n:
            raise ValueError(f"expected exactly {n} items")
        return value

    return check


def _at_most(n: int) -> Callable[[list[str]], list[str]]:
    def check(value: list[str]) -> list[str]:
        return value[:n]

    return check


Language = Annotated[str, AfterValidator(_clean_text(MAX_LANGUAGE_CHARS))]
Summary = Annotated[str, AfterValidator(_clean_text(MAX_SUMMARY_CHARS))]
Reply = Annotated[str, AfterValidator(_clean_text(MAX_REPLY_CHARS))]


class ReviewInput(BaseModel):
    review_id: str
    date: dt.date
    platform: str
    rating: float = Field(ge=1, le=5)
    review_text: str


@dataclass
class ValidationResult:
    """valid_reviews has REQUIRED_COLUMNS; rejected_rows adds a `reject_reason` column."""

    valid_reviews: pd.DataFrame
    rejected_rows: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    input_count: int = 0
    dropped_empty_count: int = 0


class ReviewAnalysis(BaseModel):
    """Structured LLM output for one review."""

    sentiment: Sentiment
    category: Category
    urgency: Urgency
    language: Language = Field(description="Language of the review, e.g. English, Tamil, Tanglish")
    summary: Summary = Field(description="One concise English sentence")
    suggested_reply: Reply = Field(description="Polite draft reply for human review")


class EnrichedReview(BaseModel):
    review_id: str
    date: dt.date
    platform: str
    rating: float
    review_text: str
    sentiment: Sentiment | None = None
    category: Category | None = None
    urgency: Urgency | None = None
    language: str | None = None
    summary: str | None = None
    suggested_reply: str | None = None
    processing_status: ProcessingStatus = "needs_review"


class WeeklyMetrics(BaseModel):
    period_start: dt.date
    period_end: dt.date
    total_reviews: int
    needs_review_count: int = 0
    sentiment_counts: dict[str, int]
    sentiment_percentages: dict[str, float]
    category_counts: dict[str, int]
    urgency_counts: dict[str, int]
    high_urgency_count: int


class WeeklyReport(BaseModel):
    top_3_complaints: Annotated[list[str], AfterValidator(_at_most(3))]
    top_praises: list[str]
    urgent_attention: str
    action_items: Annotated[list[str], AfterValidator(_exactly(3))]
