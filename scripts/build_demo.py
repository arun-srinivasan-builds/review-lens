"""Build the public demo bundle by analysing a CSV once with the real LLM.

Usage:
    python -m scripts.build_demo [input.csv] [output_dir]

Defaults: sample_data/reviews_200plus.csv -> demo_data/. Needs OPENAI_API_KEY and OPENAI_MODEL
(from .env). Makes one LLM call per review plus one for the report; prints counts only.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reviewlens.aggregation import calculate_weekly_metrics
from reviewlens.classification import classify_reviews, enriched_to_dataframe
from reviewlens.config import ConfigError, load_settings
from reviewlens.demo import save_demo_bundle
from reviewlens.llm import get_llm
from reviewlens.logging_utils import get_logger
from reviewlens.report import (
    ReportGenerationError,
    build_review_evidence,
    generate_weekly_report,
)
from reviewlens.security import safe_error_message
from reviewlens.validation import UploadError, load_reviews, validate_reviews

logger = get_logger("reviewlens.build_demo")


class _LocalFile:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.size = path.stat().st_size
        self._data = path.read_bytes()

    def getvalue(self) -> bytes:
        return self._data


def main(argv: list[str]) -> int:
    source = Path(argv[1]) if len(argv) > 1 else Path("sample_data/reviews_200plus.csv")
    target = Path(argv[2]) if len(argv) > 2 else Path("demo_data")
    try:
        settings = load_settings()
        llm = get_llm(settings)
        validation = validate_reviews(load_reviews(_LocalFile(source), settings), settings)
    except (ConfigError, UploadError, OSError) as exc:
        logger.error("cannot start: %s", safe_error_message(exc))
        return 2

    logger.info("analysing %d reviews (%d rejected)", len(validation.valid_reviews), len(validation.rejected_rows))
    fatal: list[str] = []
    items = classify_reviews(validation.valid_reviews, settings=settings, llm=llm, on_fatal_error=fatal.append)
    enriched = enriched_to_dataframe(items)
    if fatal:
        logger.error("provider rejected the requests: %s", fatal[0])
        return 3

    metrics = calculate_weekly_metrics(enriched)
    if metrics.needs_review_count:
        logger.warning("%d review(s) need review; bundle will include them", metrics.needs_review_count)
    try:
        report = generate_weekly_report(
            metrics, build_review_evidence(enriched), llm=llm, settings=settings
        )
    except ReportGenerationError as exc:
        logger.error("report failed: %s", safe_error_message(exc))
        return 4

    save_demo_bundle(target, enriched, report)
    logger.info("wrote demo bundle to %s: %d completed", target, metrics.total_reviews)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
