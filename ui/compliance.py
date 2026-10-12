"""Assignment Compliance tab: a recorded snapshot mapping the academy brief to evidence.

Source of truth: Social Eagle AI Academy "Use Case Definitions", ReviewLens (Track 1) sections 1.1-1.8,
plus "Common expectations" and "Submission" on page 16. Row numbering inside each section is ours; the
section numbers are the academy's. Nothing here is live monitoring: it is a record of one assessment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import streamlit as st

from ui.theme import escape_html, section_label

Status = Literal["Completed", "Pending", "Not Verified"]
Kind = Literal["mandatory", "deliverable", "stretch", "optional"]

ASSESSED_ON = "12 Oct 2026"
ASSESSED_COMMIT = "c7b76f7"
SOURCE = "Social Eagle AI Academy, Use Case Definitions: ReviewLens sections 1.1-1.8 and page 16"

STATUS_LABEL: dict[str, str] = {
    "Completed": "✅ Completed",
    "Pending": "⏳ Pending",
    "Not Verified": "❔ Not Verified",
}


@dataclass(frozen=True)
class Requirement:
    ref: str
    text: str
    evidence: str
    status: Status
    paths: tuple[str, ...] = ()  # repo paths cited as evidence (checked to exist by the tests)


@dataclass(frozen=True)
class Section:
    title: str
    note: str
    kind: Kind
    items: tuple[Requirement, ...]


SECTIONS: tuple[Section, ...] = (
    Section(
        "What you will build (§1.2)",
        "Mandatory functional requirements.",
        "mandatory",
        (
            Requirement(
                "1.2-1",
                "Upload a CSV of reviews (review_id, date, platform, rating, review_text)",
                "load_reviews + Streamlit uploader; size, row, header and encoding checks; tests/test_validation.py",
                "Completed",
                ("reviewlens/validation.py", "app.py", "tests/test_validation.py"),
            ),
            Requirement(
                "1.2-2",
                "Analyse every review with a LangChain chain returning a fixed structure: sentiment, "
                "category, urgency, one-line summary, polite suggested reply (plus language, per §1.6)",
                "build_review_chain returns the Pydantic ReviewAnalysis; 206 of 206 demo reviews carry all six fields",
                "Completed",
                ("reviewlens/classification.py", "reviewlens/models.py"),
            ),
            Requirement(
                "1.2-3",
                "Tamil and Tanglish reviews are understood and summarised in English",
                "124 Tamil/Tanglish demo reviews: 0 summaries contain Tamil script; 14 random summaries read "
                "manually and judged faithful (sample, not exhaustive)",
                "Completed",
                ("demo_data/enriched.csv",),
            ),
            Requirement(
                "1.2-4",
                "Weekly insights report from a second chain: top complaints, top praises, 3 action items",
                "build_report_chain / generate_weekly_report; demo report has 3 complaints, 3 praises and "
                "exactly 3 action items",
                "Completed",
                ("reviewlens/report.py", "demo_data/report.json"),
            ),
            Requirement(
                "1.2-5",
                "Dashboard: sentiment split, category chart and a HIGH-urgency-only filter",
                "Sentiment donut, category and urgency bars, HIGH toggle; tests/test_app_smoke.py and "
                "test_high_urgency_filter_equals_metric",
                "Completed",
                ("ui/dashboard.py", "tests/test_app_smoke.py", "tests/test_pipeline_integration.py"),
            ),
            Requirement(
                "1.2-6",
                "Export the enriched table as CSV so the support team can work from it in Excel",
                "export_enriched_csv: UTF-8 with BOM, formula-injection safe; tests/test_export.py; "
                "opened in Excel by the project owner",
                "Completed",
                ("reviewlens/export.py", "tests/test_export.py"),
            ),
        ),
    ),
    Section(
        "How it works (§1.3)",
        "Mandatory pipeline steps.",
        "mandatory",
        (
            Requirement(
                "1.3-1",
                "Validate: check columns, drop empty rows",
                "validate_reviews drops and counts empty text and rejects invalid rows; demo file: 211 input, "
                "206 accepted, 5 rejected (2 empty)",
                "Completed",
                ("reviewlens/validation.py", "sample_data/reviews_200plus.csv"),
            ),
            Requirement(
                "1.3-2",
                "Classify each review in batches",
                "classify_reviews runs chain.batch in chunks with a progress callback; "
                "test_progress_callback_and_concurrency",
                "Completed",
                ("reviewlens/classification.py", "tests/test_classification.py"),
            ),
            Requirement(
                "1.3-3",
                "Aggregate: count by sentiment, category, urgency",
                "calculate_weekly_metrics (pandas only); tests/test_aggregation.py",
                "Completed",
                ("reviewlens/aggregation.py", "tests/test_aggregation.py"),
            ),
            Requirement(
                "1.3-4",
                "An invalid LLM value (e.g. a category not in the list) is retried automatically, "
                "then marked 'needs review'",
                "Bounded retry then needs_review, attempts = 1 + max_retries; 22 tests in "
                "tests/test_invalid_llm_retry.py",
                "Completed",
                ("reviewlens/classification.py", "tests/test_invalid_llm_retry.py"),
            ),
        ),
    ),
    Section(
        "Tech stack and key concepts (§1.4)",
        "Mandatory building blocks named in the brief.",
        "mandatory",
        (
            Requirement(
                "1.4-1",
                "LangChain ChatPromptTemplate and LCEL (|) chains",
                "ChatPromptTemplate | llm.with_structured_output(...) in both chains",
                "Completed",
                ("reviewlens/classification.py", "reviewlens/report.py"),
            ),
            Requirement(
                "1.4-2",
                "Structured output: Pydantic model + with_structured_output()",
                "ReviewAnalysis and WeeklyReport (Literal enums) used with with_structured_output",
                "Completed",
                ("reviewlens/models.py",),
            ),
            Requirement(
                "1.4-3",
                "Batching: chain.batch() with max_concurrency",
                "config={'max_concurrency': MAX_CONCURRENCY (default 5)}; "
                "test_max_concurrency_from_settings_passed_to_batch",
                "Completed",
                ("reviewlens/classification.py", "tests/test_pipeline_integration.py"),
            ),
            Requirement(
                "1.4-4",
                "Map then reduce: per-review chain, then one report chain",
                "classify_reviews (map) then generate_weekly_report over metrics plus stratified evidence (reduce)",
                "Completed",
                ("reviewlens/classification.py", "reviewlens/report.py"),
            ),
            Requirement(
                "1.4-5",
                "LLM: OpenAI / Gemini / Claude (any one)",
                "OpenAI through langchain-openai ChatOpenAI; provider factory in reviewlens/llm.py",
                "Completed",
                ("reviewlens/llm.py",),
            ),
            Requirement(
                "1.4-6",
                "UI: Streamlit + pandas + a chart library",
                "Streamlit app, pandas aggregation, Plotly charts",
                "Completed",
                ("app.py", "requirements.txt"),
            ),
        ),
    ),
    Section(
        "Sample input and output (§1.5-1.7)",
        "Mandatory: output should look like the brief's samples.",
        "mandatory",
        (
            Requirement(
                "1.5-1",
                "Sample input: the three-row reviews_week40.csv",
                "sample_data/sample_reviews.csv holds the brief's three rows; analysed in the live app by "
                "the owner (3 of 3 completed)",
                "Completed",
                ("sample_data/sample_reviews.csv",),
            ),
            Requirement(
                "1.6-1",
                "Output A: per-review record with review_id, sentiment, category, urgency, language, "
                "summary, suggested_reply",
                "All seven fields present for every demo review (demo_data/enriched.csv)",
                "Completed",
                ("demo_data/enriched.csv",),
            ),
            Requirement(
                "1.6-2",
                "Output A match: review R-1042 is classified as urgency high",
                "In the owner's 12 Oct 2026 run the model returned medium for R-1042 (late and cold, no "
                "health risk); urgency is model judgement, so this sample does not match",
                "Pending",
            ),
            Requirement(
                "1.6-3",
                "Output B: weekly report with period, total reviews, positive/neutral/negative %, top 3 "
                "complaints, top praise, urgent count, action items",
                "Weekly report tab shows each element; percentages and urgent count come from pandas",
                "Completed",
                ("ui/dashboard.py", "reviewlens/report.py"),
            ),
            Requirement(
                "1.6-4",
                "Output B detail: per-complaint review counts and platform / time-slot hints",
                "Not produced: ReviewLens never states issue-level counts it did not calculate; category "
                "counts are shown in the chart instead",
                "Pending",
            ),
            Requirement(
                "1.7-1",
                "Allowed values: sentiment (3), category (6), urgency (3); one-sentence English summary; "
                "polite reply",
                "Pydantic Literal enums; 206 of 206 values in the allowed lists; 0 multi-sentence summaries",
                "Completed",
                ("reviewlens/models.py", "demo_data/enriched.csv"),
            ),
        ),
    ),
    Section(
        "Done checklist (§1.8)",
        "Mandatory acceptance checklist from the brief.",
        "mandatory",
        (
            Requirement(
                "1.8-1",
                "Processes a 200+ row CSV without crashing or timing out",
                "206 reviews analysed, 206 completed, 0 needs review; one run took about 2.5 minutes "
                "(12 Oct 2026, exit code 0)",
                "Completed",
                ("sample_data/reviews_200plus.csv",),
            ),
            Requirement(
                "1.8-2",
                "Every review has all 6 fields, and every value is from the allowed list",
                "206 of 206 completed reviews; sentiment, category and urgency validated against the lists",
                "Completed",
                ("demo_data/enriched.csv", "reviewlens/models.py"),
            ),
            Requirement(
                "1.8-3",
                "Tamil and Tanglish reviews get a correct English summary",
                "14 of 14 randomly sampled Tamil/Tanglish summaries judged correct; 0 of 124 contain "
                "Tamil script. Sample check, not exhaustive",
                "Completed",
                ("demo_data/enriched.csv",),
            ),
            Requirement(
                "1.8-4",
                "Report numbers match the table because they are calculated in pandas, not guessed",
                "Independent recomputation from enriched.csv equals the reported metrics; "
                "test_metrics_match_independent_recomputation",
                "Completed",
                ("reviewlens/aggregation.py", "tests/test_pipeline_integration.py"),
            ),
            Requirement(
                "1.8-5",
                "HIGH urgency filter works and the enriched CSV opens correctly in Excel",
                "Filter covered by tests; Excel opening confirmed by the project owner (Tamil text correct, "
                "formula-like row inert)",
                "Completed",
                ("ui/dashboard.py", "reviewlens/export.py"),
            ),
            Requirement(
                "1.8-6",
                "API key is loaded from a .env file and never hard-coded",
                "load_dotenv in config.py; .env is git-ignored; source scan for key literals in "
                "tests/test_security.py",
                "Completed",
                ("reviewlens/config.py", ".gitignore", "tests/test_security.py"),
            ),
        ),
    ),
    Section(
        "Common expectations (page 16)",
        "Mandatory repository and documentation expectations.",
        "mandatory",
        (
            Requirement(
                "C-1",
                "Clean repository: clear folder structure, requirements.txt, .env.example",
                "Layout documented in the README; pinned requirements.txt; .env.example with placeholders",
                "Completed",
                ("README.md", "requirements.txt", ".env.example"),
            ),
            Requirement(
                "C-2",
                "README: what the project does and how to run it",
                "README covers purpose, architecture, quick start, CSV format and configuration",
                "Completed",
                ("README.md",),
            ),
            Requirement(
                "C-3",
                "README includes a screenshot",
                "The README currently contains no screenshot",
                "Pending",
            ),
            Requirement(
                "C-4",
                "README includes one sample input with its output",
                "Sample input is shown; the matching sample output is not yet included",
                "Pending",
            ),
            Requirement(
                "C-5",
                "No secrets in code; API keys stay in .env, which is in .gitignore",
                ".gitignore lists .env; export scanned for key patterns before publishing; "
                "tests/test_security.py",
                "Completed",
                (".gitignore", "tests/test_security.py"),
            ),
            Requirement(
                "C-6",
                "Test with your own data, not only the brief's sample",
                "211-row own dataset with English, Tamil, Tanglish, injection and formula-like cases",
                "Completed",
                ("sample_data/reviews_200plus.csv",),
            ),
            Requirement(
                "C-7",
                "Show the output: the demo shows the same kind of output as the Sample Output",
                "Public demo shows the enriched table and weekly report (see 1.6-2 and 1.6-4 for sample deviations)",
                "Completed",
                ("demo_data/enriched.csv",),
            ),
        ),
    ),
    Section(
        "Submission steps (page 16)",
        "Submission deliverables.",
        "deliverable",
        (
            Requirement(
                "S-1",
                "Push your code to GitHub",
                "github.com/arun-srinivasan-builds/review-lens, branch main",
                "Completed",
            ),
            Requirement(
                "S-2",
                "Share the repository link in the WhatsApp group",
                "Owner action; no evidence available to this assessment",
                "Not Verified",
            ),
            Requirement(
                "S-3",
                "Record a short demo video and post it in the community",
                "Owner action; no evidence available to this assessment",
                "Not Verified",
            ),
            Requirement(
                "S-4",
                "Self-check: someone new can run the project in 10 minutes using only the README",
                "Not tested with a person who has not seen the project",
                "Not Verified",
            ),
        ),
    ),
    Section(
        "Stretch goals (§1.8)",
        "Optional in the brief: not required for the Done checklist.",
        "stretch",
        (
            Requirement(
                "X-1",
                "Compare this week with last week (trend arrows)",
                "Not implemented",
                "Pending",
            ),
            Requirement(
                "X-2",
                "Generate replies in the same language as the review",
                "Tamil 57 of 57 replied in Tamil, English 82 of 82 in English, Tanglish 62 of 67 in Tanglish "
                "(5 replied in Tamil script)",
                "Completed",
                ("reviewlens/classification.py", "demo_data/enriched.csv"),
            ),
            Requirement(
                "X-3",
                "Email the weekly report automatically every Monday",
                "Not implemented",
                "Pending",
            ),
        ),
    ),
    Section(
        "Optional deployment enhancements (not in the brief)",
        "Additional delivery work beyond what the academy asks for.",
        "optional",
        (
            Requirement(
                "D-1",
                "Public no-login demo using a pre-analysed sample (no AI calls)",
                "reviewlens.arunsrinivasan.io returned HTTP 200 on 12 Oct 2026; demo container has no API key",
                "Completed",
                ("docker-compose.yml",),
            ),
            Requirement(
                "D-2",
                "Password-protected live analysis site",
                "reviewlens-admin.arunsrinivasan.io returned HTTP 401 without credentials; owner signed in "
                "and analysed a file",
                "Completed",
                ("docker-compose.yml", "docs/DEPLOY.md"),
            ),
            Requirement(
                "D-3",
                "Docker image and reverse-proxy deployment runbook",
                "Image built and smoke-tested in demo and live modes; runbook in docs/DEPLOY.md",
                "Completed",
                ("Dockerfile", "docs/DEPLOY.md"),
            ),
        ),
    ),
)

ENGINEERING_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("Test suite", "Full automated test suite", "199 passed"),
    ("Reliability", "Invalid-LLM retry tests", "22 passed"),
    ("Reliability", "Retry limit and failed-review isolation", "Passed"),
    ("Accuracy", "Metrics recomputed independently", "Passed"),
    ("Code quality", "Ruff static checks", "Passed"),
    ("Safety", "Real OpenAI calls during tests (outbound network blocked)", "0"),
    ("Change control", "Production code changes needed for the retry tests", "0"),
)


def all_requirements() -> list[Requirement]:
    return [item for section in SECTIONS for item in section.items]


def status_counts(kind: Kind | None = None) -> Counter[str]:
    return Counter(
        item.status
        for section in SECTIONS
        if kind is None or section.kind == kind
        for item in section.items
    )


def _frame(section: Section) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "#": item.ref,
                "Assignment Requirement": item.text,
                "Evidence": item.evidence,
                "Status": STATUS_LABEL[item.status],
            }
            for item in section.items
        ]
    )


# ---- presentation only: group and a concise name for every mandatory requirement -----------------
GROUPS: tuple[str, ...] = (
    "CSV Processing",
    "AI Classification",
    "Weekly Insights",
    "Dashboard & Export",
    "Documentation",
)

# ref -> (group, concise requirement name). Statuses and evidence come from SECTIONS, unchanged.
PRESENTATION: dict[str, tuple[str, str]] = {
    "1.2-1": ("CSV Processing", "CSV upload with the five required columns"),
    "1.3-1": ("CSV Processing", "Validation: columns checked, empty rows dropped"),
    "1.5-1": ("CSV Processing", "Sample CSV from the brief processes end to end"),
    "1.8-1": ("CSV Processing", "200+ reviews processed without crashing or timing out"),
    "1.2-2": ("AI Classification", "LangChain chain returns a fixed structure per review"),
    "1.2-3": ("AI Classification", "Tamil and Tanglish understood, summarised in English"),
    "1.3-2": ("AI Classification", "Reviews classified in batches"),
    "1.3-4": ("AI Classification", "Invalid LLM value retried, then marked needs review"),
    "1.4-1": ("AI Classification", "ChatPromptTemplate and LCEL chains"),
    "1.4-2": ("AI Classification", "Pydantic structured output (with_structured_output)"),
    "1.4-3": ("AI Classification", "chain.batch with max_concurrency"),
    "1.4-5": ("AI Classification", "LLM provider: OpenAI (any one is allowed)"),
    "1.6-1": ("AI Classification", "Per-review record has all required fields (Output A)"),
    "1.6-2": ("AI Classification", "Sample output match: review R-1042 rated high urgency"),
    "1.7-1": ("AI Classification", "Allowed sentiment, category and urgency values enforced"),
    "1.8-2": ("AI Classification", "Every review has all 6 fields from the allowed lists"),
    "1.8-3": ("AI Classification", "Correct English summary for Tamil and Tanglish"),
    "1.2-4": ("Weekly Insights", "Weekly report from a second chain"),
    "1.3-3": ("Weekly Insights", "Counts by sentiment, category and urgency (pandas)"),
    "1.4-4": ("Weekly Insights", "Map then reduce: per-review chain, then report chain"),
    "1.6-3": ("Weekly Insights", "Report content: totals, percentages, complaints, praise, actions"),
    "1.6-4": ("Weekly Insights", "Sample output detail: per-complaint review counts"),
    "1.8-4": ("Weekly Insights", "Report numbers match the table (calculated, not guessed)"),
    "1.2-5": ("Dashboard & Export", "Dashboard: sentiment split and category chart"),
    "1.2-6": ("Dashboard & Export", "Enriched table exported as CSV for Excel"),
    "1.4-6": ("Dashboard & Export", "Streamlit, pandas and a chart library"),
    "1.8-5": ("Dashboard & Export", "HIGH-urgency filter works; CSV opens correctly in Excel"),
    "C-7": ("Dashboard & Export", "Demo shows the same kind of output as the brief"),
    "C-1": ("Documentation", "Clean repository, requirements.txt, .env.example"),
    "C-2": ("Documentation", "README: what it does and how to run it"),
    "C-3": ("Documentation", "README screenshot"),
    "C-4": ("Documentation", "README sample input with its output"),
    "C-5": ("Documentation", "No secrets in code; .env is git-ignored"),
    "C-6": ("Documentation", "Tested with own data, not only the brief's sample"),
    "1.8-6": ("Documentation", "API key loaded from .env, never hard-coded"),
}

_STATUS_HTML: dict[str, str] = {
    "Completed": '<span class="rl-st rl-ok"><span class="rl-dot">✓</span>Completed</span>',
    "Pending": '<span class="rl-st rl-warn"><span class="rl-dot">●</span>Pending</span>',
    "Not Verified": '<span class="rl-st rl-na"><span class="rl-dot">○</span>Not verified</span>',
}


def mandatory_items() -> list[Requirement]:
    return [i for s in SECTIONS if s.kind == "mandatory" for i in s.items]


def grouped_mandatory() -> dict[str, list[tuple[str, Requirement]]]:
    """Group name -> [(concise name, requirement)] in checklist order."""
    grouped: dict[str, list[tuple[str, Requirement]]] = {g: [] for g in GROUPS}
    for item in mandatory_items():
        group, short = PRESENTATION[item.ref]
        grouped[group].append((short, item))
    return grouped


def _cards_html(counts: Counter[str]) -> str:
    def card(label: str, value: int, css: str) -> str:
        return (
            f'<div class="rl-card"><div class="rl-card-label">{escape_html(label)}</div>'
            f'<div class="rl-card-num {css}">{int(value)}</div></div>'
        )

    return (
        '<div class="rl-cards">'
        + card("Completed", counts["Completed"], "rl-ok")
        + card("Pending", counts["Pending"], "rl-warn")
        + card("Not verified", counts["Not Verified"], "rl-na")
        + "</div>"
    )


def _checklist_html(rows: list[tuple[str, Requirement]]) -> str:
    body = "".join(
        f"<tr><td>{escape_html(short)}</td><td class='rl-status-cell'>{_STATUS_HTML[item.status]}</td></tr>"
        for short, item in rows
    )
    return (
        '<table class="rl-check"><thead><tr><th>Requirement</th><th>Status</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _detail_frame(section: Section) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "#": item.ref,
                "Requirement (brief wording)": item.text,
                "Evidence": item.evidence,
                "Status": STATUS_LABEL[item.status],
            }
            for item in section.items
        ]
    )


def _simple_table(section: Section) -> None:
    st.caption(section.note)
    st.table(
        pd.DataFrame(
            [
                {"Item": item.text, "Status": STATUS_LABEL[item.status], "Evidence": item.evidence}
                for item in section.items
            ]
        ).set_index("Item")
    )


def render_compliance_tab() -> None:
    """Mentor-friendly summary first; evidence, source and extras sit in expanders below."""
    st.subheader("Assignment Compliance")
    st.caption("Social Eagle AI Academy | GenAI / LangChain | ReviewLens")

    mandatory = status_counts("mandatory")
    st.markdown(_cards_html(mandatory), unsafe_allow_html=True)
    st.caption(
        f"Mandatory requirements only ({sum(mandatory.values())} items). "
        f"Recorded snapshot, assessed {ASSESSED_ON} at commit {ASSESSED_COMMIT}; not live monitoring."
    )

    for group, rows in grouped_mandatory().items():
        done = sum(1 for _, item in rows if item.status == "Completed")
        with st.container(border=True):
            section_label(f"{group}  ·  {done} of {len(rows)} completed")
            st.markdown(_checklist_html(rows), unsafe_allow_html=True)

    with st.expander("Verification Details"):
        st.markdown(f"**Source:** {SOURCE}.")
        st.markdown(f"**Assessed:** {ASSESSED_ON}  ·  **Verified commit:** `{ASSESSED_COMMIT}`")
        st.caption(
            "Requirement numbers follow the academy's section numbers (1.2 to 1.8) and page 16 "
            "(C = common expectations). Evidence was checked against the source code, tests, the analysed "
            "demo data and the live deployment."
        )
        for section in SECTIONS:
            if section.kind == "mandatory":
                section_label(section.title)
                st.table(_detail_frame(section).set_index("#"))

    for section in SECTIONS:
        if section.kind == "stretch":
            with st.expander("Stretch goals (optional in the brief)"):
                _simple_table(section)
        elif section.kind == "optional":
            with st.expander("Deployment enhancements (beyond the brief)"):
                _simple_table(section)
        elif section.kind == "deliverable":
            with st.expander("Submission steps"):
                _simple_table(section)

    with st.expander("Additional Engineering Validation"):
        st.caption(
            "Additional validation performed during the build. These are not separate academy requirements."
        )
        st.table(
            pd.DataFrame(ENGINEERING_CHECKS, columns=["Area", "Check", "Result"]).set_index("Area")
        )
        st.caption(
            "Recorded with mocked language-model responses and outbound network connections blocked, "
            "so no real API call was possible."
        )
