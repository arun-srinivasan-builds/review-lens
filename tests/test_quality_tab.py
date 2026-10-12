"""The public demo's Quality & Validation tab: static, clearly labelled, no sensitive content."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from reviewlens.demo import save_demo_bundle
from tests.test_demo import _enriched, _report
from ui.quality import RECORDED_CHECKS, recorded_frame

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _demo_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    save_demo_bundle(tmp_path, _enriched(), _report())
    monkeypatch.setenv("REVIEWLENS_MODE", "demo")
    monkeypatch.setenv("REVIEWLENS_DEMO_DIR", str(tmp_path))
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def test_recorded_results_match_the_requested_summary() -> None:
    rows = {c.check: c.result for c in RECORDED_CHECKS}
    assert rows["Full automated test suite"] == "185 passed"
    assert rows["Invalid LLM response retry tests"] == "22 passed"
    assert rows["Retry limit enforcement"] == "Passed"
    assert rows["Failed-review isolation"] == "Passed"
    assert rows["Metrics accuracy (counts and percentages)"] == "Passed"
    assert rows["Ruff static checks"] == "Passed"
    assert rows["Real OpenAI calls during tests"] == "0"
    assert rows["Production code changes required for retry tests"] == "0"
    assert list(recorded_frame().columns) == ["Area", "Check", "Result"]


def test_demo_has_quality_tab_labelled_as_a_recorded_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert labels == ["Overview", "Weekly report", "Reviews", "Quality & Validation"]  # dashboard stays first
    info = " ".join(i.value for i in at.info)
    assert "recorded validation run" in info
    assert "not live monitoring" in info


def test_quality_tab_exposes_no_secrets_or_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    text = " ".join(
        str(getattr(e, "value", ""))
        for coll in (at.markdown, at.text, at.caption, at.info, at.subheader)
        for e in coll
    ).lower()
    for forbidden in ("api_key", "sk-", "traceback", ".env", "password", "secret", "token"):
        assert forbidden not in text


def test_live_mode_has_no_quality_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.fakes import FakeChatLLM  # local import keeps the module header small

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-FAKEKEY0123456789abcdef")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setattr("reviewlens.llm.get_llm", lambda settings: FakeChatLLM())
    csv = (
        b"review_id,date,platform,rating,review_text\n"
        b"A1,2026-10-01,Google,5,Great food\n"
    )
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.file_uploader[0].set_value(("reviews.csv", csv, "text/csv"))
    at.run(timeout=60)
    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Weekly report", "Reviews"]
