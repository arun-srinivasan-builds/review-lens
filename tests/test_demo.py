"""Public demo mode: pre-analysed bundle, no uploads, no LLM."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from reviewlens.config import ConfigError, load_settings
from reviewlens.demo import DemoDataError, load_demo_bundle, save_demo_bundle
from reviewlens.models import ENRICHED_COLUMNS, WeeklyReport

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _enriched() -> pd.DataFrame:
    rows = [
        ("R1", "positive", "food_quality", "low", "completed"),
        ("R2", "negative", "delivery", "high", "completed"),
        ("R3", None, None, None, "needs_review"),
    ]
    return pd.DataFrame(
        [
            {
                "review_id": rid,
                "date": dt.date(2026, 10, 5),
                "platform": "Google",
                "rating": 3.0,
                "review_text": "=SUM(1,2) Tamil: சாப்பாடு",
                "sentiment": s,
                "category": c,
                "urgency": u,
                "language": "English" if s else None,
                "summary": "Summary." if s else None,
                "suggested_reply": "Thanks." if s else None,
                "processing_status": status,
            }
            for rid, s, c, u, status in rows
        ],
        columns=list(ENRICHED_COLUMNS),
    )


def _report() -> WeeklyReport:
    return WeeklyReport(
        top_3_complaints=["Late delivery"],
        top_praises=["Tasty food"],
        urgent_attention="Look at delivery.",
        action_items=["a", "b", "c"],
    )


def test_bundle_round_trip(tmp_path: Path) -> None:
    save_demo_bundle(tmp_path, _enriched(), _report())
    df, report = load_demo_bundle(tmp_path)
    assert list(df["review_id"]) == ["R1", "R2", "R3"]
    assert df.loc[2, "processing_status"] == "needs_review"
    assert df.loc[2, "sentiment"] is None or pd.isna(df.loc[2, "sentiment"])
    assert df.loc[0, "review_text"].startswith("=SUM")  # stored raw; export sanitises on download
    assert "சாப்பாடு" in df.loc[0, "review_text"]
    assert report is not None and len(report.action_items) == 3


def test_missing_bundle_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DemoDataError, match="build_demo"):
        load_demo_bundle(tmp_path)


def test_malformed_enum_is_rejected(tmp_path: Path) -> None:
    save_demo_bundle(tmp_path, _enriched(), None)
    path = tmp_path / "enriched.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("positive", "ecstatic"), encoding="utf-8")
    with pytest.raises(DemoDataError):
        load_demo_bundle(tmp_path)


def test_invalid_mode_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEWLENS_MODE", "admin")
    with pytest.raises(ConfigError):
        load_settings()


def test_demo_app_has_no_uploader_and_never_calls_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_demo_bundle(tmp_path, _enriched(), _report())
    monkeypatch.setenv("REVIEWLENS_MODE", "demo")
    monkeypatch.setenv("REVIEWLENS_DEMO_DIR", str(tmp_path))

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("LLM must not be constructed in demo mode")

    monkeypatch.setattr("reviewlens.llm.get_llm", _boom)
    monkeypatch.setattr("reviewlens.classification.get_llm", _boom)
    monkeypatch.setattr("reviewlens.report.build_report_chain", _boom)

    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert len(at.file_uploader) == 0
    assert not [b for b in at.button if "Analyse" in b.label]
    assert any("Pre-analysed sample: 2 reviews" in s.value for s in at.success)


def test_demo_app_without_bundle_shows_friendly_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVIEWLENS_MODE", "demo")
    monkeypatch.setenv("REVIEWLENS_DEMO_DIR", str(tmp_path / "nope"))
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("Demo data not found" in e.value for e in at.error)
