"""The public demo's Assignment Compliance tab: honest, evidence-backed and free of sensitive content."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from reviewlens.aggregation import (
    calculate_complaint_breakdown,
    calculate_weekly_metrics,
)
from reviewlens.demo import load_demo_bundle, save_demo_bundle
from reviewlens.report import format_report_text
from tests.test_demo import _enriched, _report
from ui.compliance import (
    ASSESSED_COMMIT,
    ASSESSED_ON,
    ENGINEERING_CHECKS,
    GROUPS,
    PRESENTATION,
    REASONS,
    SECTIONS,
    STATUS_LABEL,
    all_requirements,
    grouped_mandatory,
    mandatory_items,
    reason_for,
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
    assert by_ref["1.6-2"] == "Pending"  # R-1042 urgency differs from the brief
    assert by_ref["1.6-4"] == "Completed"  # counts and platform are calculated in pandas
    assert by_ref["1.6-5"] == "Pending"  # the CSV has no time of day
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
    match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    has_screenshot = bool(match) and (ROOT / match.group(1)).is_file()
    status = {r.ref: r.status for r in all_requirements()}
    # the table must say Completed only if the README really has a screenshot, and vice versa
    assert (status["C-3"] == "Completed") == has_screenshot


def test_readme_sample_output_matches_the_analysed_bundle() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "### Sample input and output" in readme
    df, report = load_demo_bundle(ROOT / "demo_data")
    assert report is not None

    # Output A: the three README records equal the bundle rows, field for field
    block = readme.split("**Output A", 1)[1].split("```json", 1)[1].split("```", 1)[0]
    records = json.loads(block)
    assert len(records) == 3
    keys = ("review_id", "sentiment", "category", "urgency", "language", "summary", "suggested_reply")
    for rec in records:
        assert tuple(rec) == keys
        row = df[df["review_id"] == rec["review_id"]].iloc[0]
        assert all(rec[k] == row[k] for k in keys)

    # Output B: figures and text equal what the bundle produces
    expected = format_report_text(
        calculate_weekly_metrics(df), report, calculate_complaint_breakdown(df, top_n=3)
    ).rstrip()
    assert expected in readme
    assert "WEEKLY FEEDBACK REPORT" in readme and "ACTION ITEMS" in readme


def test_readme_sample_output_status_is_consistent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    has_sample_output = "**Output A" in readme and "WEEKLY FEEDBACK REPORT" in readme
    status = {r.ref: r.status for r in all_requirements()}
    assert (status["C-4"] == "Completed") == has_sample_output


# ---- rendering
def test_demo_shows_compliance_tab_with_date_commit_and_snapshot_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Weekly report", "Reviews", "Assignment Compliance"]
    text = _visible_text(at)
    assert "Assignment Compliance" in text
    assert "Social Eagle AI Academy | GenAI / LangChain | ReviewLens" in text
    assert ASSESSED_ON in text
    assert ASSESSED_COMMIT in text
    assert "not live monitoring" in text
    assert "Requirement" in text and "Status" in text


def test_main_view_is_grouped_and_shows_three_mandatory_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    text = _visible_text(at)
    counts = status_counts("mandatory")
    for group in GROUPS:
        assert group.replace("&", "&amp;") in text  # labels are HTML-escaped; browsers render '&'
    assert ">Completed<" in text and ">Pending<" in text and ">Not verified<" in text
    for value in (counts["Completed"], counts["Pending"], counts["Not Verified"]):
        assert 'rl-card-num' in text and f">{value}<" in text
    # every pending mandatory item stays individually visible in the main checklist
    for name in (
        "Sample output match: review R-1042 rated high urgency",
        "Sample output detail: time-slot hint (e.g. 8pm to 10pm)",
    ):
        assert name in text
    assert text.count("Pending</span>") >= counts["Pending"]


def test_details_and_extras_live_in_expanders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    labels = [e.label for e in at.expander]
    for expected in (
        "Verification Details",
        "Stretch goals (optional in the brief)",
        "Deployment enhancements (beyond the brief)",
        "Submission steps",
        "Additional Engineering Validation",
    ):
        assert expected in labels
    # evidence and commit detail are inside "Verification Details", not in the main checklist
    details = next(e for e in at.expander if e.label == "Verification Details")
    detail_text = " ".join(str(getattr(m, "value", "")) for m in details.markdown) + " ".join(
        t.value.to_string() for t in details.table
    )
    assert ASSESSED_COMMIT in detail_text
    assert "Evidence" in detail_text


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


# ---- presentation guarantees: concise names, five groups, statuses untouched
def test_every_mandatory_requirement_is_mapped_to_one_of_five_groups() -> None:
    mandatory = {i.ref for i in mandatory_items()}
    assert set(PRESENTATION) == mandatory  # no item hidden, merged away or invented
    assert {g for g, _ in PRESENTATION.values()} == set(GROUPS)
    assert GROUPS == (
        "CSV Processing",
        "AI Classification",
        "Weekly Insights",
        "Dashboard & Export",
        "Documentation",
    )
    assert sum(len(rows) for rows in grouped_mandatory().values()) == len(mandatory)


def test_grouping_does_not_change_any_status() -> None:
    by_ref = {r.ref: r.status for r in all_requirements()}
    shown = {item.ref: item.status for rows in grouped_mandatory().values() for _, item in rows}
    assert shown == {ref: by_ref[ref] for ref in shown}
    pending = sorted(ref for ref, st in shown.items() if st == "Pending")
    assert pending == ["1.6-2", "1.6-5"]  # exactly the known pending mandatory items


def test_concise_names_are_short_and_unique() -> None:
    names = [short for _, short in PRESENTATION.values()]
    assert len(names) == len(set(names))
    assert all(0 < len(n) <= 75 for n in names)


def test_readme_screenshot_is_a_real_png_of_the_dashboard() -> None:
    path = ROOT / "docs" / "screenshot.png"
    data = path.read_bytes()
    assert data[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])  # genuine PNG signature
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width >= 600 and height >= 400
    assert 20_000 < len(data) < 2_000_000  # large enough to be a screenshot, small enough for a README
    assert "docs/screenshot.png" in (ROOT / "README.md").read_text(encoding="utf-8")


# ---- Reason column: pending items are explained, completed items show an em dash
def test_every_non_completed_requirement_has_a_reason() -> None:
    missing = [r.ref for r in all_requirements() if r.status != "Completed" and not reason_for(r).strip()]
    assert missing == []


def test_completed_requirements_show_an_em_dash_reason() -> None:
    completed = [r for r in all_requirements() if r.status == "Completed"]
    assert completed and all(reason_for(r) == "\u2014" for r in completed)
    assert not any(ref in REASONS for ref in (r.ref for r in completed))  # no stale reasons kept


def test_r1042_reason_is_the_agreed_wording() -> None:
    assert REASONS["1.6-2"] == (
        "The AI classified the late and cold delivery as medium urgency. The academy's illustrative "
        "sample labels it high. Classification was not overridden solely to match the sample."
    )


def test_time_slot_reason_states_the_csv_has_no_times() -> None:
    reason = REASONS["1.6-5"].lower()
    assert "no order or delivery time" in reason
    assert "not invented" in reason


def test_reason_column_is_shown_in_verification_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _demo_app(tmp_path, monkeypatch)
    details = next(e for e in at.expander if e.label == "Verification Details")
    frames = [t.value for t in details.table]
    assert frames and all("Reason" in f.columns and "Evidence" in f.columns for f in frames)
    weekly = next(f for f in frames if "1.6-2" in f.index)
    assert "medium urgency" in weekly.loc["1.6-2", "Reason"]
    completed_rows = weekly[weekly["Status"].str.contains("Completed")]
    assert (completed_rows["Reason"] == "\u2014").all()
