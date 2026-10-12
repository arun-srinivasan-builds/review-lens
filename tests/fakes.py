"""Shared offline fakes for integration, security and app tests (no network, ever)."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from reviewlens.models import (
    CATEGORIES,
    SENTIMENTS,
    URGENCIES,
    ReviewAnalysis,
    WeeklyReport,
)

FAKE_KEY = "sk-test-FAKEKEY0123456789abcdef"
PLAIN_FAKE_KEY = "FAKE_PLAIN_KEY_0123456789"  # does not match the sk- redaction pattern


def deterministic_analysis(text: str) -> ReviewAnalysis:
    """Pure function of the review text so tests can recompute expectations independently."""
    h = zlib.crc32(text.encode("utf-8"))
    return ReviewAnalysis(
        sentiment=SENTIMENTS[h % 3],  # type: ignore[arg-type]
        category=CATEGORIES[(h // 3) % 6],  # type: ignore[arg-type]
        urgency=URGENCIES[(h // 18) % 3],  # type: ignore[arg-type]
        language="Tamil" if any("஀" <= c <= "௿" for c in text) else "English",
        summary=f"Summary of review {h % 1000}.",
        suggested_reply="Thank you for your feedback.",
    )


def fake_report() -> WeeklyReport:
    return WeeklyReport(
        top_3_complaints=["Late delivery", "Cold food", "App crashes"],
        top_praises=["Tasty biryani"],
        urgent_attention="Investigate high-urgency complaints today.",
        action_items=["Review delivery partners", "Check food temperature", "Fix app payment crash"],
    )


def human_text(prompt_value: Any) -> str:
    return str(prompt_value.to_messages()[-1].content)


def system_text(prompt_value: Any) -> str:
    return str(prompt_value.to_messages()[0].content)


class FakeChatLLM:
    """Stand-in for BaseChatModel. `review_handler(text)` and `report_handler()` produce outputs."""

    def __init__(
        self,
        review_handler: Callable[[str], Any] | None = None,
        report_handler: Callable[[], Any] | None = None,
    ) -> None:
        self.review_handler = review_handler or (lambda text: deterministic_analysis(text))
        self.report_handler = report_handler or fake_report
        self.review_prompts: list[Any] = []
        self.report_prompts: list[Any] = []

    def with_structured_output(self, schema: Any) -> Runnable:
        if schema is ReviewAnalysis:

            def run_review(prompt_value: Any) -> Any:
                self.review_prompts.append(prompt_value)
                text = human_text(prompt_value).removeprefix("<review>\n").removesuffix("\n</review>")
                return self.review_handler(text)

            return RunnableLambda(run_review)

        def run_report(prompt_value: Any) -> Any:
            self.report_prompts.append(prompt_value)
            return self.report_handler()

        return RunnableLambda(run_report)
