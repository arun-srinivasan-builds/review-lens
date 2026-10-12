"""Public demo mode: serve a pre-analysed dataset and report without any LLM calls."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from reviewlens.classification import enriched_to_dataframe
from reviewlens.models import ENRICHED_COLUMNS, EnrichedReview, WeeklyReport

ENRICHED_FILE = "enriched.csv"
REPORT_FILE = "report.json"
_OPTIONAL_TEXT = ("sentiment", "category", "urgency", "language", "summary", "suggested_reply")


class DemoDataError(Exception):
    """The demo bundle is missing or malformed. Messages contain no review text."""


def _row_to_review(row: dict[str, str], line: int) -> EnrichedReview:
    data: dict[str, object] = {k: row.get(k, "") for k in ENRICHED_COLUMNS}
    for key in _OPTIONAL_TEXT:
        if data[key] == "":
            data[key] = None
    try:
        data["date"] = dt.date.fromisoformat(str(data["date"]))
        data["rating"] = float(str(data["rating"]))
        return EnrichedReview.model_validate(data)
    except (ValueError, ValidationError) as exc:
        raise DemoDataError(f"Demo data row {line} is invalid.") from exc


def load_demo_bundle(directory: Path) -> tuple[pd.DataFrame, WeeklyReport | None]:
    """Load enriched.csv (+ optional report.json) written by scripts/build_demo.py."""
    enriched_path = directory / ENRICHED_FILE
    if not enriched_path.is_file():
        raise DemoDataError(
            f"Demo data not found ({ENRICHED_FILE}). Run `python -m scripts.build_demo` to create it."
        )
    try:
        raw = pd.read_csv(enriched_path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DemoDataError(f"Demo data ({ENRICHED_FILE}) could not be read.") from exc
    missing = [c for c in ENRICHED_COLUMNS if c not in raw.columns]
    if missing:
        raise DemoDataError(f"Demo data is missing column(s): {', '.join(missing)}.")
    if raw.empty:
        raise DemoDataError("Demo data contains no reviews.")

    reviews = [_row_to_review(rec, i + 2) for i, rec in enumerate(raw.to_dict("records"))]
    enriched = enriched_to_dataframe(reviews)

    report: WeeklyReport | None = None
    report_path = directory / REPORT_FILE
    if report_path.is_file():
        try:
            report = WeeklyReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError) as exc:
            raise DemoDataError(f"Demo report ({REPORT_FILE}) is invalid.") from exc
    return enriched, report


def save_demo_bundle(directory: Path, enriched: pd.DataFrame, report: WeeklyReport | None) -> None:
    """Write the bundle. CSV is stored raw (UTF-8); the app sanitises again on download."""
    directory.mkdir(parents=True, exist_ok=True)
    enriched[list(ENRICHED_COLUMNS)].to_csv(directory / ENRICHED_FILE, index=False, encoding="utf-8")
    if report is not None:
        (directory / REPORT_FILE).write_text(report.model_dump_json(indent=2), encoding="utf-8")
