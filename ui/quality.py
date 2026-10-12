"""Quality & Validation tab: a static record of a verified test run (not live monitoring)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from ui.theme import section_label

RECORDED_ON = "12 Oct 2026"
RECORDED_COMMIT = "c924068"


@dataclass(frozen=True)
class RecordedCheck:
    area: str
    check: str
    result: str


RECORDED_CHECKS: tuple[RecordedCheck, ...] = (
    RecordedCheck("Test suite", "Full automated test suite", "185 passed"),
    RecordedCheck("Reliability", "Invalid LLM response retry tests", "22 passed"),
    RecordedCheck("Reliability", "Retry limit enforcement", "Passed"),
    RecordedCheck("Reliability", "Failed-review isolation", "Passed"),
    RecordedCheck("Accuracy", "Metrics accuracy (counts and percentages)", "Passed"),
    RecordedCheck("Code quality", "Ruff static checks", "Passed"),
    RecordedCheck("Safety", "Real OpenAI calls during tests", "0"),
    RecordedCheck("Change control", "Production code changes required for retry tests", "0"),
)


def recorded_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Area": c.area, "Check": c.check, "Result": c.result} for c in RECORDED_CHECKS]
    )


def render_quality_tab() -> None:
    """Compact table of recorded results. Contains no configuration, logs or secrets."""
    st.subheader("Quality & Validation")
    st.info(
        f"Results from a recorded validation run on {RECORDED_ON} (commit {RECORDED_COMMIT}). "
        "This is a snapshot, not live monitoring."
    )
    with st.container(border=True):
        section_label("Recorded validation results")
        st.dataframe(
            recorded_frame(),
            width="stretch",
            hide_index=True,
            column_config={
                "Area": st.column_config.TextColumn("Area", width="small"),
                "Check": st.column_config.TextColumn("Check", width="large"),
                "Result": st.column_config.TextColumn("Result", width="small"),
            },
        )
    st.caption(
        "All tests use mocked language-model responses. This run was recorded with outbound network "
        "connections blocked, so no real API call was possible."
    )
