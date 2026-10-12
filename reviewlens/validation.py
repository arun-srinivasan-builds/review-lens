"""Module 1: CSV upload checks and row-level validation. Never logs review text."""

from __future__ import annotations

import datetime as dt
import io
import math
import re
import warnings
from collections import Counter
from typing import Any

import pandas as pd

from reviewlens.config import Settings, load_settings
from reviewlens.logging_utils import get_logger
from reviewlens.models import REQUIRED_COLUMNS, ValidationResult
from reviewlens.security import sanitize_text

logger = get_logger(__name__)

_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$")
_MAX_EXAMPLES = 3
_MAX_ID_CHARS = 64
_MAX_PLATFORM_CHARS = 50


class UploadError(Exception):
    """Raised when an uploaded file cannot be processed. Messages are user-friendly and secret-free."""


def _read_bytes(file: Any) -> bytes:
    """Get raw bytes from a Streamlit UploadedFile or any file-like/bytes-like object."""
    try:
        if hasattr(file, "getvalue"):
            data = file.getvalue()
        elif hasattr(file, "read"):
            data = file.read()
        else:
            raise UploadError("The uploaded file could not be read. Please upload a CSV file.")
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError("The uploaded file could not be read. Please try uploading it again.") from exc
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not isinstance(data, (bytes, bytearray)):
        raise UploadError("The uploaded file could not be read. Please upload a CSV file.")
    return bytes(data)


def load_reviews(file: Any, settings: Settings | None = None) -> pd.DataFrame:
    """Check and parse an uploaded CSV into an all-string DataFrame (no row validation)."""
    settings = settings or load_settings()

    name = str(getattr(file, "name", "") or "")
    if not name.lower().endswith(".csv"):
        raise UploadError("Unsupported file type. Please upload a file with a .csv extension.")

    declared_size = getattr(file, "size", None)
    if isinstance(declared_size, int) and declared_size > settings.max_upload_bytes:
        raise UploadError(
            f"File is too large ({declared_size / (1024 * 1024):.1f} MB). "
            f"The maximum allowed size is {settings.max_upload_mb} MB."
        )
    raw = _read_bytes(file)
    if len(raw) > settings.max_upload_bytes:
        raise UploadError(
            f"File is too large ({len(raw) / (1024 * 1024):.1f} MB). "
            f"The maximum allowed size is {settings.max_upload_mb} MB."
        )
    if not raw.strip():
        raise UploadError("The uploaded file is empty. Please upload a CSV with a header row and reviews.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadError(
            "The file is not valid UTF-8. Please re-save it as 'CSV UTF-8' and upload it again."
        ) from exc

    text = text.replace("\0", "")  # NUL bytes would silently truncate fields in the parser
    try:
        with warnings.catch_warnings():
            # rows with more fields than the header would otherwise be truncated silently
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(
                io.StringIO(text), dtype=str, keep_default_na=False, index_col=False, on_bad_lines="error"
            )
    except pd.errors.EmptyDataError as exc:
        raise UploadError("The uploaded file has no data. Please include a header row and reviews.") from exc
    except (pd.errors.ParserError, pd.errors.ParserWarning, ValueError) as exc:
        raise UploadError(
            "The CSV could not be parsed. Check for unbalanced quotes or rows with extra columns."
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise UploadError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected headers: {', '.join(REQUIRED_COLUMNS)}."
        )
    df = df[list(REQUIRED_COLUMNS)].copy()  # drop extra columns (may hold unneeded personal data)
    if len(df) > settings.max_rows:
        raise UploadError(
            f"Too many rows ({len(df)}). The maximum allowed is {settings.max_rows} rows per upload."
        )
    if len(df) == 0:
        raise UploadError("The CSV has a header but no review rows.")

    logger.info("Loaded CSV upload with %d rows", len(df))
    return df


def _parse_date(value: str) -> dt.date | None:
    candidate = value.strip()
    if not _ISO_DATE.match(candidate):
        return None
    parsed = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_rating(value: str) -> float | None:
    try:
        rating = float(value.strip())
    except ValueError:
        return None
    if math.isnan(rating) or math.isinf(rating) or not 1 <= rating <= 5:
        return None
    return rating


def validate_reviews(df: pd.DataFrame, settings: Settings | None = None) -> ValidationResult:
    """Validate rows; invalid rows go to rejected_rows with a reject_reason (never silently accepted)."""
    settings = settings or load_settings()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise UploadError(f"Missing required column(s): {', '.join(missing)}.")

    seen_ids: set[str] = set()
    valid_records: list[dict[str, Any]] = []
    reject_flags: list[tuple[int, str]] = []  # (positional index, reason)
    dropped_empty = 0

    for pos, (_, row) in enumerate(df.iterrows()):
        review_id = sanitize_text(str(row["review_id"]))
        platform = sanitize_text(str(row["platform"]))
        raw_text = str(row["review_text"])
        text = sanitize_text(raw_text)

        if not text:
            dropped_empty += 1
            reject_flags.append((pos, "empty review_text"))
            continue

        reasons: list[str] = []
        if not review_id:
            reasons.append("blank review_id")
        elif review_id in seen_ids:
            reasons.append("duplicate review_id")
        parsed_date = _parse_date(str(row["date"]))
        if parsed_date is None:
            reasons.append("unparseable date (expected YYYY-MM-DD)")
        rating = _parse_rating(str(row["rating"]))
        if rating is None:
            reasons.append("rating must be a number from 1 to 5")
        if not platform:
            reasons.append("blank platform")
        elif len(platform) > _MAX_PLATFORM_CHARS:
            reasons.append(f"platform longer than {_MAX_PLATFORM_CHARS} characters")
        if len(review_id) > _MAX_ID_CHARS:
            reasons.append(f"review_id longer than {_MAX_ID_CHARS} characters")
        if len(text) > settings.max_review_chars:
            reasons.append(f"review_text longer than {settings.max_review_chars} characters")

        if reasons:
            reject_flags.append((pos, "; ".join(reasons)))
            continue

        seen_ids.add(review_id)
        valid_records.append(
            {
                "review_id": review_id,
                "date": parsed_date,
                "platform": platform,
                "rating": rating,
                "review_text": text,
            }
        )

    valid = pd.DataFrame(valid_records, columns=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if not valid.empty:
        valid["rating"] = valid["rating"].astype(float)

    rejected = df.iloc[[p for p, _ in reject_flags]].copy().reset_index(drop=True)
    rejected["reject_reason"] = [reason for _, reason in reject_flags]

    errors = _summarise(reject_flags)
    logger.info(
        "Validated %d rows: %d valid, %d rejected (%d empty)",
        len(df),
        len(valid),
        len(rejected),
        dropped_empty,
    )
    return ValidationResult(
        valid_reviews=valid,
        rejected_rows=rejected,
        errors=errors,
        input_count=len(df),
        dropped_empty_count=dropped_empty,
    )


def _summarise(flags: list[tuple[int, str]]) -> list[str]:
    """Human-readable lines grouped by reason, citing CSV line numbers (header is line 1)."""
    if not flags:
        return []
    grouped: dict[str, list[int]] = {}
    for pos, reason in flags:
        for single in reason.split("; "):
            grouped.setdefault(single, []).append(pos + 2)
    counts = Counter({reason: len(lines) for reason, lines in grouped.items()})
    lines_out: list[str] = []
    for reason, count in counts.most_common():
        examples = ", ".join(str(n) for n in grouped[reason][:_MAX_EXAMPLES])
        more = " ..." if count > _MAX_EXAMPLES else ""
        lines_out.append(f"{count} row(s) rejected: {reason} (CSV line(s) {examples}{more})")
    return lines_out
