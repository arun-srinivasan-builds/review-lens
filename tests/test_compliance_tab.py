"""The public demo's Assignment Compliance tab: honest, evidence-backed and free of sensitive content."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from reviewlens.demo import load_demo_bundle, save_demo_bundle
from tests.test_demo import _enriched, _report
from ui.compliance import (
    ASSESSED_COMMIT,
    ASSESSED_ON,
    ENGINEERING_CHECKS,
    SECTIONS,
    STATUS_LABEL,
    all_requirements,
    status_counts,
)

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(ROOT / "app.py")
TAMIL = re.compile("[஀-௿]")


def _demo_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    save_demo_bundle(tmp_path, _enriched(), _report())
    monkeypatch.setenv("REVIEWLENS_MODE", "demo")
    monkeypatch.setenv("REVIEWLENS_DEMO_DIR", str(tmp_path))
    return AppTest.from_file(APP_PATH, default_timeout=60).run()


def _visible_text(at: AppTest) -> str:
    parts: list[str] = []
    for coll in (at.markdown, at.text, at.caption, at.info, at.subheader, at.metric):
        parts += [str(getattr(e, "value", "")) for e in coll]
    for table in at.table:
        parts.append(table.value.to_string())
    return " ".join(parts)


# ---- structure and honesty of the data
def test_only_the_three_allowed_statuses_are_used() -> None:
    assert {r.status for r in all_requirements()} <= {"Completed", "Pending", "Not Verified"}
    assert set(STATUS_LABEL) == {"Completed", "Pending", "Not Verified"}


def test_mandatory_deliverable_stretch_and_optional_are_separate_sections() -> None:
    kinds = {s.kind for s in SECTIONS}
    assert kinds == {"mandatory", "deliverable", "stretch", "optional"}
    stretch_refs = [i.ref for s in SECTIONS if s.kind == "stretch" for i in s.items]
    assert stretch_refs and all(ref.startswith("X-") for ref in stretch_refs)
    optional_refs = [i.ref for s in SECTIONS if s.kind == "optional" for i in s.items]
    assert optional_refs and all(ref.startswith("D-") for ref in optional_refs)


def test_every_requirement_has_evidence_and_refs_are_unique() -> None:
    reqs = all_requirements()
    assert all(r.evidence.strip() and r.text.strip() for r in reqs)
    refs = [r.ref for r in reqs]
    assert len(refs) == len(set(refs))


def test_cited_repo_paths_exist() -> None:
    missing = [(r.ref, p) for r in all_requirements() for p in r.paths if not (ROOT / p).exists()]
    assert missing == []


def test_every_academy_checklist_item_is_present() -> None:
    refs = {r.ref for r in all_requirements()}
    assert {f"1.8-{n}" for n in range(1, 7)} <= refs  # the six Done-checklist items
    assert {f"1.2-{n}" for n in range(1, 7)} <= refs  # the six "what you will build" items
    assert {"S-1", "S-2", "S-3"} <= refs  # GitHub, WhatsApp, demo video


def test_known_gaps_are_not_marked_completed() -> None:
    by_ref = {r.ref: r.status for r in all_requirements()}
    assert by_ref["C-3"] == "Pending"  # README screenshot
    assert by_ref["C-4"] == "Pending"  # README sample output
    assert by_ref["1.6-2"] == "Pending"  # R-1042 urgency differs from the brief
    assert by_ref["1.6-4"] == "Pending"  # per-complaint counts not produced
    assert by_ref["X-1"] == "Pending" and by_ref["X-3"] == "Pending"  # unimplemented stretch goals
    assert by_ref["S-2"] == "Not Verified" and by_ref["S-3"] == "Not Verified"  # owner actions


def test_mandatory_counts_match_the_data() -> None:
    counts = status_counts("mandatory")
    mandatory = [i for s in SECTIONS if s.kind == "mandatory" for i in s.items]
    assert sum(counts.values()) == len(mandatory)
    assert counts["Completed"] == sum(1 for i in mandatory if i.status == "Completed")


def test_engineering_validation_is_labelled_as_additional() -> None:
    labels = {c[1] for c in ENGINEERING_CHECKS}
    assert "Invalid-LLM retry tests" in labels
    results = {c[1]: c[2] for c in ENGINEERING_CHECKS}
    assert results["Invalid-LLM retry tests"] == "22 passed"
    assert results["Real OpenAI calls during tests (outbound network blocked)"] == "0"


# ---- claims about the recorded demo bundle are re-checked against the real files
def test_demo_bundle_claims_hold() -> None:
    df, report = load_demo_bundle(ROOT / "demo_data")
    done = df[df["processing_status"] == "completed"]
    assert len(df) == 206 and len(done) == 206  # 1.8-1: 200+ rows, all completed

    non_english = done[done["language"].isin(["Tamil", "Tanglish"])]
    assert len(non_english) == 124  # 1.2-3
    assert not non_english["summary"].map(lambda s: bool(TAMIL.search(str(s)))).any()

    six = ["sentiment", "category", "urgency", "language", "summary", "suggested_reply"]
    assert (done[six].astype(str).apply(lambda c: c.str.strip() != "")).all().all()  # 1.8-2
    assert report is not None and len(report.action_items) == 3  # 1.2-4

    # X-2 stretch claim: reply language follows the review language
    tamil_reply = lambda s: bool(TAMIL.search(str(s)))
    english = done[done["language"] == "English"]
    tamil = done[done["language"] == "Tamil"]
    tanglish = done[done["language"] == "Tanglish"]
    assert (len(english), int((~english["suggested_reply"].map(tamil_reply)).sum())) == (82, 82)
    assert (len(tamil), int(tamil["suggested_reply"].map(tamil_reply).sum())) == (57, 57)
    assert (len(tanglish), int((~tanglish["suggested_reply"].map(tamil_reply)).sum())) == (67, 62)


def test_readme_gaps_reported_in_the_table_are_real() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    has_screenshot = bool(re.search(r"!\[[^\]]*\]\([^)]+\)", readme))
    status = {r.ref: r.status for r in all_requirements()}
    # the table must say Completed only if the README really has a screenshot, and vice versa
    assert (status["C-3"] == "Completed") == has_screenshot


# ---- rendering
def test_demo_shows_compliance_tab_with_date_commit_and_snapshot_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Weekly report", "Reviews", "Assignment Compliance"]
    text = _visible_text(at)
    assert ASSESSED_ON in text
    assert ASSESSED_COMMIT in text
    assert "not live monitoring" in text
    for header in ("Assignment Requirement", "Evidence", "Status"):
        assert header in text
    assert "Additional Engineering Validation" in [e.label for e in at.expander]


def test_compliance_tab_exposes_no_secrets_or_infrastructure_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    text = _visible_text(at).lower()
    for forbidden in ("sk-", "traceback", "admin_password_hash", "$2a$", "187.127", "openai_api_key="):
        assert forbidden not in text


def test_live_mode_has_no_compliance_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.fakes import FakeChatLLM

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-FAKEKEY0123456789abcdef")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setattr("reviewlens.llm.get_llm", lambda settings: FakeChatLLM())
    csv = (b"review_id,date,platform,rating,review_text\nA1,2026-10-01,Google,5,Great food\n")
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    at.file_uploader[0].set_value(("reviews.csv", csv, "text/csv"))
    at.run(timeout=60)
    at.button[0].click()
    at.run(timeout=60)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Weekly report", "Reviews"]
