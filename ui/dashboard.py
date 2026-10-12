"""Dashboard and weekly report rendering. All user and model text is rendered as plain text."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reviewlens.export import export_enriched_csv
from reviewlens.models import (
    CATEGORIES,
    SENTIMENTS,
    URGENCIES,
    WeeklyMetrics,
    WeeklyReport,
)
from reviewlens.report import format_report_text
from ui.compliance import render_compliance_tab
from ui.theme import section_label

SENTIMENT_COLOURS = {"positive": "#2F9E6F", "neutral": "#94A3B8", "negative": "#DC2626"}
URGENCY_COLOURS = {"low": "#94A3B8", "medium": "#F59E0B", "high": "#DC2626"}
CATEGORY_COLOUR = "#4338CA"
INK = "#334155"

SENTIMENT_BADGE = {"positive": "🟢 Positive", "neutral": "⚪ Neutral", "negative": "🔴 Negative"}
URGENCY_BADGE = {"low": "🟢 Low", "medium": "🟠 Medium", "high": "🔴 High"}
DRAFT_LABEL = "Draft for human review — not sent"
REPLY_COLUMN = f"Suggested reply ({DRAFT_LABEL.lower()})"

TABLE_COLUMNS = [
    "review_id",
    "date",
    "platform",
    "rating",
    "review_text",
    "sentiment",
    "category",
    "urgency",
    "summary",
    "suggested_reply",
    "processing_status",
]


def _pretty(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _style(fig: go.Figure, height: int = 310) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"size": 13, "color": INK},
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        height=height,
        showlegend=False,
    )
    return fig


def _sentiment_chart(metrics: WeeklyMetrics) -> go.Figure:
    labels = [k for k in SENTIMENTS if k in metrics.sentiment_counts]
    values = [metrics.sentiment_counts[k] for k in labels]
    fig = go.Figure(
        go.Pie(
            labels=[_pretty(k) for k in labels],
            values=values,
            hole=0.62,
            sort=False,
            marker={"colors": [SENTIMENT_COLOURS[k] for k in labels], "line": {"color": "#FFFFFF", "width": 2}},
            textinfo="percent",
            hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{metrics.total_reviews}</b><br><span style='font-size:12px;color:#64748B'>reviews</span>",
        showarrow=False,
        font={"size": 26, "color": "#0F172A"},
    )
    return _style(fig)


def _bar_chart(counts: dict[str, int], order: tuple[str, ...], colours: dict[str, str] | str) -> go.Figure:
    items = sorted(((k, counts.get(k, 0)) for k in order), key=lambda kv: kv[1])
    colour = [colours[k] for k, _ in items] if isinstance(colours, dict) else colours
    peak = max((v for _, v in items), default=0)
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in items],
            y=[_pretty(k) for k, _ in items],
            orientation="h",
            marker={"color": colour},
            text=[v for _, v in items],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x} reviews<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, max(peak * 1.18, 1)])
    fig.update_yaxes(showgrid=False, ticks="")
    return _style(fig)


def _kpis(metrics: WeeklyMetrics) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", metrics.total_reviews, border=True)
    c2.metric("Needs review", metrics.needs_review_count, border=True)
    c3.metric("High urgency", metrics.high_urgency_count, border=True)
    c4.metric(
        "Period",
        f"{metrics.period_start.strftime('%d %b')} – {metrics.period_end.strftime('%d %b %Y')}",
        border=True,
    )


def _urgent_panel(enriched_df: pd.DataFrame) -> None:
    high = enriched_df[enriched_df["urgency"] == "high"]
    with st.container(border=True):
        section_label("Needs attention first")
        if high.empty:
            st.caption("No high-urgency reviews in this period.")
            return
        for _, row in high.sort_values("review_id").head(6).iterrows():
            st.text(f"{row['review_id']}  ·  {row['platform']}  ·  {_pretty(str(row['category']))}  —  {row['summary']}")
        if len(high) > 6:
            st.caption(f"+ {len(high) - 6} more in the Reviews tab (filter: HIGH urgency only).")


def _overview(enriched_df: pd.DataFrame, metrics: WeeklyMetrics) -> None:
    _kpis(metrics)
    st.write("")
    c1, c2, c3 = st.columns(3)
    chart_config = {"displayModeBar": False}
    with c1.container(border=True):
        section_label("Sentiment")
        st.plotly_chart(_sentiment_chart(metrics), width="stretch", config=chart_config)
    with c2.container(border=True):
        section_label("Categories")
        st.plotly_chart(
            _bar_chart(metrics.category_counts, CATEGORIES, CATEGORY_COLOUR), width="stretch", config=chart_config
        )
    with c3.container(border=True):
        section_label("Urgency")
        st.plotly_chart(
            _bar_chart(metrics.urgency_counts, URGENCIES, URGENCY_COLOURS), width="stretch", config=chart_config
        )
    _urgent_panel(enriched_df)


def _report_tab(metrics: WeeklyMetrics, report: WeeklyReport | None) -> None:
    st.subheader("Weekly feedback report")
    pct = metrics.sentiment_percentages
    with st.container(border=True):
        section_label("Verified figures (calculated, not AI-written)")
        a, b, c, d = st.columns(4)
        a.metric("Period", f"{metrics.period_start.isoformat()} to {metrics.period_end.isoformat()}")
        b.metric("Total reviews analysed", metrics.total_reviews)
        c.metric(
            "Sentiment",
            f"{pct.get('positive', 0.0):.1f}% positive",
            f"{pct.get('negative', 0.0):.1f}% negative",
            delta_color="inverse",
        )
        d.metric("URGENT", f"{metrics.high_urgency_count} high-urgency review(s)")
    if report is None:
        st.info("The narrative report is unavailable. The verified metrics above are still correct.")
        return

    st.caption("The narrative below is AI-written from the analysed reviews; the counts above are calculated.")
    left, right = st.columns(2)
    with left.container(border=True):
        section_label("TOP 3 COMPLAINTS")
        for i, item in enumerate(report.top_3_complaints, start=1):
            st.text(f"{i}. {item}")
    with right.container(border=True):
        section_label("TOP PRAISE")
        for item in report.top_praises:
            st.text(f"•  {item}")
    with st.container(border=True):
        section_label("URGENT ATTENTION")
        st.text(report.urgent_attention)
    with st.container(border=True):
        section_label("ACTION ITEMS")
        for i, item in enumerate(report.action_items, start=1):
            st.text(f"{i}. {item}")
    st.download_button(
        "Download report (.txt)",
        data=format_report_text(metrics, report).encode("utf-8"),
        file_name="reviewlens_weekly_report.txt",
        mime="text/plain",
        on_click="ignore",
    )


def _filter_controls(enriched_df: pd.DataFrame) -> pd.DataFrame:
    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2])
    query = c1.text_input("Search", key="review_search", placeholder="Review text, summary or ID").strip()
    sentiments = c2.multiselect("Sentiment", [_pretty(s) for s in SENTIMENTS], key="filter_sentiment")
    categories = c3.multiselect("Category", [_pretty(c) for c in CATEGORIES], key="filter_category")
    urgencies = c4.multiselect("Urgency", [_pretty(u) for u in URGENCIES], key="filter_urgency")
    high_only = st.toggle("Show HIGH urgency only", value=False, key="high_urgency_only")

    view = enriched_df.copy()
    if query:
        haystack = (
            view["review_text"].fillna("").astype(str)
            + " "
            + view["summary"].fillna("").astype(str)
            + " "
            + view["review_id"].fillna("").astype(str)
        ).str.lower()
        view = view[haystack.str.contains(query.lower(), regex=False)]
    if sentiments:
        view = view[view["sentiment"].isin([s.lower() for s in sentiments])]
    if categories:
        view = view[view["category"].isin([c.lower().replace(" ", "_") for c in categories])]
    if urgencies:
        view = view[view["urgency"].isin([u.lower() for u in urgencies])]
    if high_only:
        # needs_review rows have no urgency yet, so keep them visible rather than hiding possible urgent cases
        view = view[(view["urgency"] == "high") | (view["processing_status"] == "needs_review")]
    return view


def _display_frame(view: pd.DataFrame) -> pd.DataFrame:
    out = view[[c for c in TABLE_COLUMNS if c in view.columns]].copy()
    out["sentiment"] = out["sentiment"].map(lambda v: SENTIMENT_BADGE.get(v, ""))
    out["urgency"] = out["urgency"].map(lambda v: URGENCY_BADGE.get(v, ""))
    out["category"] = out["category"].map(lambda v: _pretty(v) if isinstance(v, str) else "")
    out["processing_status"] = out["processing_status"].map(
        lambda s: "NEEDS REVIEW" if s == "needs_review" else "Completed"
    )
    return out.rename(
        columns={
            "review_id": "ID",
            "date": "Date",
            "platform": "Platform",
            "rating": "Rating",
            "review_text": "Review",
            "sentiment": "Sentiment",
            "category": "Category",
            "urgency": "Urgency",
            "summary": "Summary (English)",
            "suggested_reply": REPLY_COLUMN,
            "processing_status": "Status",
        }
    )


def _reviews_tab(enriched_df: pd.DataFrame) -> None:
    st.caption(f"Suggested replies: {DRAFT_LABEL}.")
    view = _filter_controls(enriched_df)
    shown = _display_frame(view)

    def _flag(row: pd.Series) -> list[str]:
        style = "background-color: #FFF7E0; color: #000000" if row.get("Status") == "NEEDS REVIEW" else ""
        return [style] * len(row)

    st.dataframe(
        shown.style.apply(_flag, axis=1),
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Rating": st.column_config.NumberColumn("Rating", format="%.0f ★", width="small"),
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Platform": st.column_config.TextColumn("Platform", width="small"),
            "Review": st.column_config.TextColumn("Review", width="large"),
            "Summary (English)": st.column_config.TextColumn("Summary (English)", width="large"),
            REPLY_COLUMN: st.column_config.TextColumn(REPLY_COLUMN, width="large"),
        },
    )
    left, right = st.columns([3, 1])
    left.caption(f"Showing {len(view)} of {len(enriched_df)} reviews.")
    right.download_button(
        "Download enriched CSV",
        data=export_enriched_csv(enriched_df),
        file_name="reviewlens_enriched.csv",
        mime="text/csv",
        on_click="ignore",
        width="stretch",
    )


def render_dashboard(
    enriched_df: pd.DataFrame,
    metrics: WeeklyMetrics,
    report: WeeklyReport | None,
    show_compliance: bool = False,
) -> None:
    """Render overview, weekly report and the filterable review table with CSV download.

    `show_compliance` adds a static "Assignment Compliance" tab (used by the public demo only).
    """
    st.header("Dashboard")
    names = ["Overview", "Weekly report", "Reviews"] + (["Assignment Compliance"] if show_compliance else [])
    tabs = st.tabs(names)
    with tabs[0]:
        _overview(enriched_df, metrics)
    with tabs[1]:
        _report_tab(metrics, report)
    with tabs[2]:
        _reviews_tab(enriched_df)
    if show_compliance:
        with tabs[3]:
            render_compliance_tab()
