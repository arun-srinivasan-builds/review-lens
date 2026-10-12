"""End-to-end pipeline with a fake LLM: CSV -> validate -> classify -> metrics -> report -> export."""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langchain_core.runnables import Runnable

from reviewlens import classification
from reviewlens.aggregation import calculate_weekly_metrics
from reviewlens.classification import (
    build_review_chain,
    classify_reviews,
    enriched_to_dataframe,
)
from reviewlens.config import Settings
from reviewlens.export import export_enriched_csv
from reviewlens.models import (
    CATEGORIES,
    ENRICHED_COLUMNS,
    SENTIMENTS,
    URGENCIES,
    WeeklyReport,
)
from reviewlens.report import build_review_evidence, generate_weekly_report
from reviewlens.validation import load_reviews, validate_reviews
from tests.fakes import FakeChatLLM, deterministic_analysis, human_text, system_text

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "reviews_200plus.csv"
FORMULA = re.compile(r"^\s*[=+\-@]")


class _Upload(io.BytesIO):
    name = "reviews_200plus.csv"


@pytest.fixture(scope="module")
def validated() -> Any:
    settings = Settings(openai_model="fake")
    df = load_reviews(_Upload(SAMPLE.read_bytes()), settings)
    return validate_reviews(df, settings)


@pytest.fixture
def failing_texts(validated: Any) -> set[str]:
    texts = validated.valid_reviews["review_text"].tolist()
    return {texts[i] for i in range(3, len(texts), 35)}


class _Counting:
    """Records per-text call counts; returns an invalid enum (a dict) for `failing` texts."""

    def __init__(self, failing: set[str]) -> None:
        self.failing = failing
        self.calls: Counter[str] = Counter()

    def __call__(self, text: str) -> Any:
        self.calls[text] += 1
        if text in self.failing:
            return {**deterministic_analysis(text).model_dump(), "sentiment": "meh"}
        return deterministic_analysis(text)


@pytest.fixture
def pipeline(validated: Any, failing_texts: set[str], fake_settings: Settings) -> dict[str, Any]:
    counter = _Counting(failing_texts)
    llm = FakeChatLLM(review_handler=counter)
    items = classify_reviews(validated.valid_reviews, fake_settings, llm=llm)  # type: ignore[arg-type]
    enriched = enriched_to_dataframe(items)
    return {
        "counter": counter,
        "llm": llm,
        "items": items,
        "enriched": enriched,
        "metrics": calculate_weekly_metrics(enriched),
        "settings": fake_settings,
        "valid": validated.valid_reviews,
    }


def test_sample_has_over_200_valid_rows(validated: Any) -> None:
    assert len(validated.valid_reviews) >= 200
    assert len(validated.rejected_rows) >= 5
    assert validated.input_count == len(validated.valid_reviews) + len(validated.rejected_rows)


def test_all_rows_processed_in_order(pipeline: dict[str, Any]) -> None:
    valid, enriched = pipeline["valid"], pipeline["enriched"]
    assert len(enriched) == len(valid) >= 200
    assert enriched["review_id"].tolist() == valid["review_id"].tolist()
    assert list(enriched.columns) == list(ENRICHED_COLUMNS)


def test_completed_rows_have_all_six_fields_with_allowed_values(pipeline: dict[str, Any]) -> None:
    done = pipeline["enriched"].query("processing_status == 'completed'")
    assert len(done) > 150
    for col in ("sentiment", "category", "urgency", "language", "summary", "suggested_reply"):
        assert done[col].notna().all(), col
        assert (done[col].astype(str).str.strip() != "").all(), col
    assert set(done["sentiment"]) <= set(SENTIMENTS)
    assert set(done["category"]) <= set(CATEGORIES)
    assert set(done["urgency"]) <= set(URGENCIES)
    assert "high" in set(done["urgency"])


def test_needs_review_rows_keep_source_and_are_excluded(pipeline: dict[str, Any], failing_texts: set[str]) -> None:
    enriched, valid, metrics = pipeline["enriched"], pipeline["valid"], pipeline["metrics"]
    nr = enriched[enriched["processing_status"] == "needs_review"]
    assert len(nr) == int(valid["review_text"].isin(failing_texts).sum()) > 0
    src = valid.set_index("review_id")
    for _, row in nr.iterrows():
        assert row["review_text"] == src.loc[row["review_id"], "review_text"]
        assert row["platform"] == src.loc[row["review_id"], "platform"]
        assert pd.isna(row["sentiment"]) and pd.isna(row["summary"])
    assert metrics.needs_review_count == len(nr)
    assert metrics.total_reviews == len(enriched) - len(nr)
    assert sum(metrics.sentiment_counts.values()) == metrics.total_reviews
    assert sum(metrics.category_counts.values()) == metrics.total_reviews
    assert sum(metrics.urgency_counts.values()) == metrics.total_reviews


def test_metrics_match_independent_recomputation(pipeline: dict[str, Any], failing_texts: set[str]) -> None:
    valid, metrics = pipeline["valid"], pipeline["metrics"]
    ok_texts = [t for t in valid["review_text"] if t not in failing_texts]
    expected = [deterministic_analysis(t) for t in ok_texts]
    assert metrics.total_reviews == len(expected)
    assert metrics.sentiment_counts == {s: sum(e.sentiment == s for e in expected) for s in SENTIMENTS}
    assert metrics.category_counts == {c: sum(e.category == c for e in expected) for c in CATEGORIES}
    assert metrics.urgency_counts == {u: sum(e.urgency == u for e in expected) for u in URGENCIES}
    assert metrics.high_urgency_count == sum(e.urgency == "high" for e in expected)
    for s in SENTIMENTS:
        assert metrics.sentiment_percentages[s] == round(metrics.sentiment_counts[s] / len(expected) * 100, 1)
    assert abs(sum(metrics.sentiment_percentages.values()) - 100) <= 0.2
    done = pipeline["enriched"].query("processing_status == 'completed'")
    assert metrics.sentiment_counts == done["sentiment"].value_counts().reindex(list(SENTIMENTS), fill_value=0).to_dict()
    dates = pd.to_datetime(pipeline["enriched"]["date"])
    assert (metrics.period_start, metrics.period_end) == (dates.min().date(), dates.max().date())


def test_high_urgency_filter_equals_metric(pipeline: dict[str, Any]) -> None:
    enriched, metrics = pipeline["enriched"], pipeline["metrics"]
    assert len(enriched[enriched["urgency"] == "high"]) == metrics.high_urgency_count > 0


def test_call_counts_bounded(pipeline: dict[str, Any], failing_texts: set[str]) -> None:
    cfg: Settings = pipeline["settings"]
    counter: _Counting = pipeline["counter"]
    dup_counts = Counter(pipeline["valid"]["review_text"])
    for text, n in counter.calls.items():
        per_row = n / dup_counts[text]
        assert per_row == (1 + cfg.max_retries if text in failing_texts else 1), text


def test_prompt_injection_is_only_data(pipeline: dict[str, Any]) -> None:
    llm: FakeChatLLM = pipeline["llm"]
    injections = [p for p in llm.review_prompts if "ignore previous instructions" in human_text(p).lower()]
    assert injections, "sample must contain injection reviews"
    for p in injections:
        human = human_text(p)
        assert human.startswith("<review>\n") and human.endswith("\n</review>")
        assert "ignore previous instructions" not in system_text(p).lower()
        assert len(p.to_messages()) == 2
    enriched = pipeline["enriched"]
    inj = enriched[enriched["review_text"].str.contains("Ignore previous instructions", case=False)]
    assert len(inj) >= 2
    for _, row in inj[inj["processing_status"] == "completed"].iterrows():
        assert row["urgency"] == deterministic_analysis(row["review_text"]).urgency


def test_report_and_action_items(pipeline: dict[str, Any]) -> None:
    llm = FakeChatLLM()
    enriched = pipeline["enriched"]
    evidence = build_review_evidence(enriched)
    nr_ids = enriched.query("processing_status == 'needs_review'")["review_id"]
    assert evidence
    assert not any(line.startswith(f"[{rid} ") for line in evidence for rid in nr_ids)
    report = generate_weekly_report(pipeline["metrics"], evidence, llm=llm, settings=pipeline["settings"])  # type: ignore[arg-type]
    assert isinstance(report, WeeklyReport)
    assert len(report.action_items) == 3
    assert len(llm.report_prompts) == 1
    assert f'"total_reviews": {pipeline["metrics"].total_reviews}' in human_text(llm.report_prompts[0])


def test_export_round_trip(pipeline: dict[str, Any]) -> None:
    data = export_enriched_csv(pipeline["enriched"])
    assert data.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(io.StringIO(data.decode("utf-8-sig")), dtype=str, keep_default_na=False)
    assert list(back.columns) == list(ENRICHED_COLUMNS)
    assert len(back) == len(pipeline["enriched"])
    src = pipeline["valid"].set_index("review_id")["review_text"]
    assert any(re.search("[஀-௿]", t) for t in src), "sample must contain Tamil script"
    out = back.set_index("review_id")["review_text"]
    for rid, original in src.items():
        expected = "'" + original if FORMULA.match(original) else original
        assert out[rid] == expected, rid
    assert sum(bool(FORMULA.match(t)) for t in src) >= 4
    assert not any(FORMULA.match(t) for t in back["review_text"])
    assert not any(FORMULA.match(t) for col in ("summary", "suggested_reply") for t in back[col])
    assert (back.query("processing_status == 'needs_review'")["sentiment"] == "").all()


class _ChainStub(Runnable):  # type: ignore[type-arg]
    """Delegates to a real LCEL chain but lets a test intercept batch calls."""

    def __init__(self, real: Runnable, batch_hook: Any) -> None:  # type: ignore[type-arg]
        self.real = real
        self.batch_hook = batch_hook

    def invoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
        return self.real.invoke(input, config=config, **kw)

    def batch(self, inputs: Any, config: Any = None, **kw: Any) -> Any:
        self.batch_hook(inputs, config)
        return self.real.batch(inputs, config=config, **kw)


def test_failed_chunk_keeps_previous_results(
    validated: Any, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Settings(openai_api_key=fake_settings.openai_api_key, openai_model="fake", max_concurrency=2, max_retries=1)
    batches: list[int] = []

    def hook(inputs: Any, config: Any) -> None:
        batches.append(len(inputs))
        if len(batches) == 2:
            raise RuntimeError("whole chunk failed")

    real = build_review_chain(llm=FakeChatLLM())  # type: ignore[arg-type]
    chain = _ChainStub(real, hook)
    monkeypatch.setattr(classification, "build_review_chain", lambda llm=None, settings=None: chain)
    df = validated.valid_reviews.head(12)
    progress: list[tuple[int, int]] = []
    items = classify_reviews(df, cfg, progress_callback=lambda d, t: progress.append((d, t)))
    assert batches[0] == 4  # chunk = 2 * concurrency
    statuses = [i.processing_status for i in items]
    assert len(items) == 12
    assert statuses[:4] == ["completed"] * 4
    # A non-retryable whole-chunk failure marks only that chunk needs_review; the rest is untouched.
    assert statuses[4:8] == ["needs_review"] * 4
    assert all(i.review_text for i in items[4:8])
    assert statuses[8:] == ["completed"] * 4
    assert progress[-1] == (12, 12)
    assert [i.review_id for i in items] == df["review_id"].tolist()


def test_max_concurrency_from_settings_passed_to_batch(
    validated: Any, fake_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    chain = _ChainStub(build_review_chain(llm=FakeChatLLM()), lambda i, c: seen.append(c))  # type: ignore[arg-type]
    monkeypatch.setattr(classification, "build_review_chain", lambda llm=None, settings=None: chain)
    classify_reviews(validated.valid_reviews.head(10), fake_settings)
    assert fake_settings.max_concurrency == 4
    assert len(seen) == 2  # 10 rows in chunks of 8
    assert all(c["max_concurrency"] == 4 for c in seen)
