"""Dashboard and weekly report rendering. All user text is rendered as plain text."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reviewlens.export import export_enriched_csv
from reviewlens.models import WeeklyMetrics, WeeklyReport

SENTIMENT_COLOURS = {"positive": "#2E8B57", "neutral": "#8A94A6", "negative": "#D64545"}
CATEGORY_COLOUR = "#3B6FB6"
TABLE_COLUMNS = [
    "review_id",
    "date",
    "platform",
    "review_text",
    "sentiment",
    "category",
    "urgency",
    "summary",
    "suggested_reply",
    "processing_status",
]
DRAFT_LABEL = "Draft for human review — not sent"


def _sentiment_chart(metrics: WeeklyMetrics) -> go.Figure:
    labels = list(metrics.sentiment_counts.keys())
    values = [metrics.sentiment_counts[k] for k in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker={"colors": [SENTIMENT_COLOURS.get(k, "#999999") for k in labels]},
            sort=False,
        )
    )
    fig.update_layout(title="Sentiment split", margin={"t": 50, "b": 10, "l": 10, "r": 10}, height=320)
    return fig


def _category_chart(metrics: WeeklyMetrics) -> go.Figure:
    items = sorted(metrics.category_counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[k.replace("_", " ") for k, _ in items],
            orientation="h",
            marker={"color": CATEGORY_COLOUR},
        )
    )
    fig.update_layout(
        title="Category breakdown",
        xaxis_title="Reviews",
        margin={"t": 50, "b": 10, "l": 10, "r": 10},
        height=320,
    )
    return fig


def _render_report(metrics: WeeklyMetrics, report: WeeklyReport | None) -> None:
    st.subheader("Weekly feedback report")
    st.write(f"**Period:** {metrics.period_start.isoformat()} to {metrics.period_end.isoformat()}")
    st.write(f"**Total reviews analysed:** {metrics.total_reviews}")
    pct = metrics.sentiment_percentages
    st.write(
        "**Sentiment:** "
        f"positive {pct.get('positive', 0.0):.1f}% | "
        f"neutral {pct.get('neutral', 0.0):.1f}% | "
        f"negative {pct.get('negative', 0.0):.1f}%"
    )
    st.write(f"**URGENT:** {metrics.high_urgency_count} high-urgency review(s)")
    if report is None:
        st.info("The narrative report is unavailable. The verified metrics above are still correct.")
        return
    st.markdown("**TOP 3 COMPLAINTS**")
    for i, item in enumerate(report.top_3_complaints, start=1):
        st.text(f"{i}. {item}")
    st.markdown("**TOP PRAISE**")
    for item in report.top_praises:
        st.text(f"- {item}")
    st.caption("The narrative below is AI-written from the analysed reviews; the counts above are calculated.")
    st.markdown("**URGENT ATTENTION**")
    st.text(report.urgent_attention)
    st.markdown("**ACTION ITEMS**")
    for i, item in enumerate(report.action_items, start=1):
        st.text(f"{i}. {item}")


def render_dashboard(
    enriched_df: pd.DataFrame, metrics: WeeklyMetrics, report: WeeklyReport | None
) -> None:
    """Render metrics, charts, report, filterable table and CSV download."""
    st.header("Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", metrics.total_reviews)
    c2.metric("Needs review", metrics.needs_review_count)
    c3.metric("High urgency", metrics.high_urgency_count)
    c4.metric("Period", f"{metrics.period_start.isoformat()} to {metrics.period_end.isoformat()}")

    left, right = st.columns(2)
    left.plotly_chart(_sentiment_chart(metrics), width="stretch")
    right.plotly_chart(_category_chart(metrics), width="stretch")

    _render_report(metrics, report)

    st.subheader("Reviews")
    st.caption(f"Suggested replies: {DRAFT_LABEL}.")
    high_only = st.toggle("Show HIGH urgency only", value=False, key="high_urgency_only")
    view = enriched_df.copy()
    if high_only and "urgency" in view.columns:
        # needs_review rows have no urgency yet, so keep them visible rather than hiding possible urgent cases
        view = view[(view["urgency"] == "high") | (view["processing_status"] == "needs_review")]
        if metrics.needs_review_count:
            st.caption(
                f"{metrics.needs_review_count} unclassified review(s) are shown too; "
                "they are not included in the urgent count."
            )
    cols = [c for c in TABLE_COLUMNS if c in view.columns]
    view = view[cols].copy()
    if "processing_status" in view.columns:
        view["processing_status"] = view["processing_status"].map(
            lambda s: "NEEDS REVIEW" if s == "needs_review" else s
        )
    if "suggested_reply" in view.columns:
        view = view.rename(columns={"suggested_reply": "suggested_reply (draft, not sent)"})
    view = view.rename(columns={"processing_status": "status"})

    def _flag(row: pd.Series) -> list[str]:
        style = "background-color: #FFF3CD; color: #000000" if row.get("status") == "NEEDS REVIEW" else ""
        return [style] * len(row)

    st.dataframe(view.style.apply(_flag, axis=1), width="stretch", hide_index=True)
    st.caption(f"Showing {len(view)} of {len(enriched_df)} reviews.")

    st.download_button(
        "Download enriched CSV",
        data=export_enriched_csv(enriched_df),
        file_name="reviewlens_enriched.csv",
        mime="text/csv",
        on_click="ignore",
    )
