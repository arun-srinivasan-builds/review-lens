"""ReviewLens - Streamlit app: Upload -> Analyse -> Dashboard/Report -> Export."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from reviewlens.aggregation import calculate_weekly_metrics
from reviewlens.classification import classify_reviews, enriched_to_dataframe
from reviewlens.config import ConfigError, Settings, load_settings
from reviewlens.demo import DemoDataError, load_demo_bundle
from reviewlens.llm import get_llm
from reviewlens.logging_utils import get_logger
from reviewlens.report import (
    ReportGenerationError,
    build_review_evidence,
    generate_weekly_report,
)
from reviewlens.security import safe_error_message
from reviewlens.validation import (
    UploadError,
    ValidationResult,
    load_reviews,
    validate_reviews,
)
from ui.dashboard import render_dashboard
from ui.theme import apply_theme, render_header, section_label, steps

logger = get_logger("reviewlens.app")

st.set_page_config(page_title="ReviewLens", page_icon=":mag:", layout="wide")
apply_theme()


def _secrets(settings: Settings | None) -> list[str]:
    return [settings.openai_api_key.get_secret_value()] if settings else []


def _sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("Settings")
        st.write(f"Max upload size: {settings.max_upload_mb} MB")
        st.write(f"Max rows: {settings.max_rows}")
        st.write(f"Max review length: {settings.max_review_chars} characters")
        st.write(f"Concurrency: {settings.max_concurrency}")
        key_ok = bool(settings.openai_api_key.get_secret_value())
        st.write(f"OPENAI_API_KEY configured: {'yes' if key_ok else 'no'}")
        st.write(f"OPENAI_MODEL configured: {'yes' if settings.openai_model else 'no'}")


def _reset_results() -> None:
    for key in ("enriched_df", "metrics", "report", "report_error"):
        st.session_state.pop(key, None)


def _show_validation(filename: str, result: ValidationResult) -> None:
    accepted = len(result.valid_reviews)
    rejected = len(result.rejected_rows)
    st.subheader("Upload summary")
    st.text(f"File: {filename}")
    cols = st.columns(4)
    cols[0].metric("Input rows", result.input_count, border=True)
    cols[1].metric("Accepted", accepted, border=True)
    cols[2].metric("Rejected", rejected, border=True)
    cols[3].metric("Dropped (empty text)", result.dropped_empty_count, border=True)
    for message in result.errors:
        st.text(f"Warning: {message}")
    if rejected:
        with st.expander(f"Rejected rows ({rejected})"):
            st.dataframe(result.rejected_rows, width="stretch", hide_index=True)


def _start_analysis() -> None:
    st.session_state["analysing"] = True
    st.session_state["notices"] = []


def _notice(message: str, detail: str = "") -> None:
    st.session_state.setdefault("notices", []).append((message, detail))


def _show_notices() -> None:
    for message, detail in st.session_state.get("notices", []):
        st.error(message)
        if detail:
            st.text(detail)


def _analyse(settings: Settings, valid_reviews: pd.DataFrame) -> None:
    try:
        llm = get_llm(settings)
    except ConfigError as exc:
        _notice(f"Configuration problem: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - constructor errors must not surface a traceback
        _notice("Could not initialise the language model.", safe_error_message(exc, _secrets(settings)))
        return

    fatal_messages: list[str] = []

    progress = st.progress(0.0, text="Classifying reviews...")

    def _on_progress(done: int, total: int) -> None:
        progress.progress(min(done / max(total, 1), 1.0), text=f"Classified {done} of {total} reviews")

    try:
        items = classify_reviews(
            valid_reviews,
            settings=settings,
            llm=llm,
            progress_callback=_on_progress,
            on_fatal_error=fatal_messages.append,
        )
        enriched_df = enriched_to_dataframe(items)
    except Exception as exc:  # noqa: BLE001 - surfaced as a safe message
        logger.error("classification failed: %s", type(exc).__name__)
        _notice("Classification failed.", safe_error_message(exc, _secrets(settings)))
        return
    finally:
        progress.empty()

    if fatal_messages:
        _notice(
            "The language model rejected the requests, so the remaining reviews were not sent. "
            "Check OPENAI_API_KEY, OPENAI_MODEL and your OpenAI quota.",
            fatal_messages[0],
        )

    # Classification results are kept even if later steps fail.
    st.session_state["enriched_df"] = enriched_df
    st.session_state.pop("metrics", None)
    st.session_state.pop("report", None)
    st.session_state.pop("report_error", None)

    try:
        metrics = calculate_weekly_metrics(enriched_df)
    except Exception as exc:  # noqa: BLE001
        logger.error("aggregation failed: %s", type(exc).__name__)
        _notice("Could not calculate metrics.", safe_error_message(exc, _secrets(settings)))
        return
    st.session_state["metrics"] = metrics

    try:
        with st.spinner("Generating weekly report..."):
            evidence = build_review_evidence(enriched_df)
            st.session_state["report"] = generate_weekly_report(
                metrics, evidence, llm=llm, settings=settings
            )
    except ReportGenerationError as exc:
        st.session_state["report"] = None
        st.session_state["report_error"] = safe_error_message(exc, _secrets(settings))
    except Exception as exc:  # noqa: BLE001
        logger.error("report failed: %s", type(exc).__name__)
        st.session_state["report"] = None
        st.session_state["report_error"] = safe_error_message(exc, _secrets(settings))


def _render_demo(settings: Settings) -> None:
    """Public demo: pre-analysed sample data. No upload, no API key, no LLM calls."""
    render_header("demo", "Customer feedback intelligence: sentiment, urgent issues and action items in minutes.")
    st.info(
        "This page shows results that were generated once from synthetic sample reviews. "
        "Nothing you do here is sent to OpenAI or stored."
    )
    try:
        enriched_df, report = load_demo_bundle(Path(settings.demo_dir))
    except DemoDataError as exc:
        st.error(str(exc))
        return
    completed = int((enriched_df["processing_status"] == "completed").sum())
    st.success(f"Pre-analysed sample: {completed} reviews.")
    render_dashboard(enriched_df, calculate_weekly_metrics(enriched_df), report, show_compliance=True)


def main() -> None:
    try:
        mode_settings = load_settings()
    except ConfigError as exc:
        st.title("ReviewLens")
        st.error(f"Configuration problem: {exc}")
        return
    if mode_settings.app_mode == "demo":
        _render_demo(mode_settings)
        return

    render_header(
        "live",
        "Turn a week of customer reviews into sentiment, categories, urgent issues and action items.",
    )
    st.info(
        "Privacy: review text is sent to the configured LLM provider (OpenAI) for processing. "
        "Suggested replies are drafts for human review and are never sent automatically. "
        "Uploads are kept in memory for this session only."
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"Configuration problem: {exc}")
        return
    _sidebar(settings)

    if "enriched_df" in st.session_state:
        steps(2)
    elif "validation" in st.session_state:
        steps(1)
    else:
        steps(0)

    with st.container(border=True):
        section_label("Upload weekly reviews (CSV)")
        uploaded = st.file_uploader(
            "Upload a reviews CSV",
            type=["csv"],
            max_upload_size=settings.max_upload_mb,
            label_visibility="collapsed",
        )
        st.caption(
            "Required columns: review_id, date, platform, rating, review_text. "
            f"Up to {settings.max_rows} rows and {settings.max_upload_mb} MB."
        )
    if uploaded is None:
        return

    upload_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get("upload_id") != upload_id:
        _reset_results()
        st.session_state["upload_id"] = upload_id
        st.session_state.pop("validation", None)

    if "validation" not in st.session_state:
        try:
            df = load_reviews(uploaded, settings)
            st.session_state["validation"] = validate_reviews(df, settings)
        except UploadError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error("Could not read the file.")
            st.text(safe_error_message(exc, _secrets(settings)))
            return

    result: ValidationResult = st.session_state["validation"]
    _show_validation(uploaded.name, result)

    if len(result.valid_reviews) == 0:
        st.error("No valid reviews to analyse.")
        return

    has_results = "enriched_df" in st.session_state
    label = "Re-run analysis (uses the API again)" if has_results else "Analyse reviews"
    st.button(
        label,
        type="primary",
        on_click=_start_analysis,
        disabled=bool(st.session_state.get("analysing")),
    )
    if st.session_state.get("analysing"):
        try:
            _analyse(settings, result.valid_reviews)
        finally:
            st.session_state["analysing"] = False
            st.rerun()

    _show_notices()

    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is None:
        return

    completed = int((enriched_df["processing_status"] == "completed").sum())
    needs_review = int((enriched_df["processing_status"] == "needs_review").sum())
    st.success(f"Analysis finished: {completed} completed, {needs_review} need review.")

    if st.session_state.get("report_error"):
        st.warning(
            "The weekly report could not be generated; metrics and reviews are still available."
        )
        st.text(st.session_state["report_error"])

    metrics = st.session_state.get("metrics")
    if metrics is not None:
        render_dashboard(enriched_df, metrics, st.session_state.get("report"))


main()
