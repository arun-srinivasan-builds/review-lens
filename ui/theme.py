"""Static visual polish. Only fixed strings are injected, never user or model text."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
  /* chrome: keep the sidebar toggle, drop deploy/menu clutter */
  #MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stStatusWidget"] { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }

  .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1320px; }

  h1 { font-weight: 750; letter-spacing: -0.025em; margin-bottom: 0.15rem; }
  h2, h3 { font-weight: 650; letter-spacing: -0.015em; }

  /* KPI cards */
  [data-testid="stMetric"] { padding: 0.35rem 0.25rem; }
  [data-testid="stMetricLabel"] p {
    text-transform: uppercase; letter-spacing: 0.07em; font-size: 0.72rem; font-weight: 600; color: #64748B;
  }
  [data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -0.02em; font-size: 1.9rem; }
  [data-testid="stMetricValue"] div { text-overflow: clip; }

  /* tabs */
  [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid #E2E8F0; }
  [data-baseweb="tab"] { font-weight: 600; padding: 0.55rem 0.9rem; }

  /* tables and inputs */
  [data-testid="stDataFrame"] { border: 1px solid #E2E8F0; border-radius: 0.6rem; overflow: hidden; }
  [data-testid="stFileUploaderDropzone"] { border-radius: 0.75rem; border: 1.5px dashed #A5B4FC; background: #F8FAFF; }

  /* static tables (compliance): readable wrapped text */
  [data-testid="stTable"] table { font-size: 0.86rem; }
  [data-testid="stTable"] td, [data-testid="stTable"] th { vertical-align: top; }


  /* assignment compliance */
  .rl-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.9rem; margin: 0.4rem 0 0.35rem 0; }
  .rl-card { border: 1px solid #E2E8F0; border-radius: 0.7rem; padding: 0.85rem 1.1rem; background: #FFFFFF; }
  .rl-card-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #64748B; }
  .rl-card-num { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; color: #0F172A; }
  .rl-check { width: 100%; border-collapse: collapse; font-size: 0.93rem; }
  .rl-check th { text-align: left; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: #64748B; padding: 0.3rem 0.4rem; border-bottom: 1px solid #E2E8F0; }
  .rl-check td { padding: 0.5rem 0.4rem; border-bottom: 1px solid #F1F5F9; vertical-align: middle; }
  .rl-check tr:last-child td { border-bottom: none; }
  .rl-check .rl-status-cell { width: 11rem; white-space: nowrap; }
  .rl-st { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; font-weight: 600; }
  .rl-dot { display: inline-block; width: 1.15rem; height: 1.15rem; line-height: 1.15rem; text-align: center; border-radius: 999px; font-size: 0.7rem; }
  .rl-ok { color: #15803D; }
  .rl-ok .rl-dot { background: #DCFCE7; }
  .rl-warn { color: #B45309; }
  .rl-warn .rl-dot { background: #FEF3C7; }
  .rl-na { color: #64748B; }
  .rl-na .rl-dot { background: #F1F5F9; }
  .rl-card-num.rl-ok { color: #15803D; }
  .rl-card-num.rl-warn { color: #B45309; }
  .rl-card-num.rl-na { color: #475569; }
  @media (max-width: 640px) { .rl-cards { grid-template-columns: 1fr; } }

  /* buttons */
  .stButton > button, .stDownloadButton > button { border-radius: 0.55rem; font-weight: 600; padding: 0.5rem 1.1rem; }

  /* brand lockup */
  .rl-tagline { color: #64748B; font-size: 1.02rem; margin: 0 0 0.9rem 0; }
  .rl-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; vertical-align: middle; margin-left: 0.6rem;
  }
  .rl-badge-demo { background: #E0E7FF; color: #3730A3; }
  .rl-badge-live { background: #DCFCE7; color: #166534; }
  .rl-steps { color: #94A3B8; font-size: 0.82rem; letter-spacing: 0.04em; margin-bottom: 1.1rem; }
  .rl-section { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #64748B; margin: 0.2rem 0 0.4rem 0; }
</style>
"""

_BADGES = {"demo": ("Public demo", "rl-badge-demo"), "live": ("Live analysis", "rl-badge-live")}


def apply_theme() -> None:
    """Inject the shared stylesheet (a constant; contains no dynamic content)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(mode: str, tagline: str) -> None:
    """Title, mode badge and tagline. `mode` must be 'demo' or 'live'; anything else shows no badge."""
    st.title("ReviewLens")
    label, css_class = _BADGES.get(mode, ("", ""))
    if label:
        st.markdown(f'<span class="rl-badge {css_class}">{label}</span>', unsafe_allow_html=True)
    st.markdown(f'<p class="rl-tagline">{_escape(tagline)}</p>', unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="rl-section">{_escape(text)}</div>', unsafe_allow_html=True)


def steps(active: int) -> None:
    names = ["Upload", "Analyse", "Review", "Export"]
    parts = [f"<b>{n}</b>" if i == active else n for i, n in enumerate(names)]
    st.markdown(f'<div class="rl-steps">{" &nbsp;›&nbsp; ".join(parts)}</div>', unsafe_allow_html=True)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def escape_html(value: str) -> str:
    """Public helper for the few static HTML fragments built from trusted constants."""
    return _escape(value)
