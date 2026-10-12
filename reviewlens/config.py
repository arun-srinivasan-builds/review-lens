"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import SecretStr


class ConfigError(Exception):
    """Raised for missing or invalid configuration. Messages never contain secrets."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "openai"
    openai_api_key: SecretStr = field(default=SecretStr(""), repr=False)
    openai_model: str = ""
    openai_temperature: float | None = None  # None = provider default (safe for all models)
    max_upload_mb: int = 10
    max_rows: int = 5000
    max_review_chars: int = 2000
    max_concurrency: int = 5
    llm_timeout_seconds: int = 60
    max_retries: int = 2
    app_mode: str = "live"  # "live" (upload + OpenAI) or "demo" (pre-analysed, no LLM)
    demo_dir: str = "demo_data"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}.")
    return value


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number.") from exc
    if not 0 <= value <= 2:
        raise ConfigError(f"{name} must be between 0 and 2.")
    return value


def _mode_env() -> str:
    mode = os.environ.get("REVIEWLENS_MODE", "live").strip().lower() or "live"
    if mode not in ("live", "demo"):
        raise ConfigError("REVIEWLENS_MODE must be 'live' or 'demo'.")
    return mode


def load_settings() -> Settings:
    """Load settings from the environment (and .env). Does not validate LLM credentials."""
    load_dotenv()
    # Review data must only go to the selected LLM provider: never to a tracing service.
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        os.environ[name] = "false"
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "openai").strip().lower() or "openai",
        openai_api_key=SecretStr(os.environ.get("OPENAI_API_KEY", "").strip()),
        openai_model=os.environ.get("OPENAI_MODEL", "").strip(),
        openai_temperature=_optional_float_env("OPENAI_TEMPERATURE"),
        max_upload_mb=_int_env("MAX_UPLOAD_MB", 10),
        max_rows=_int_env("MAX_ROWS", 5000),
        max_review_chars=_int_env("MAX_REVIEW_CHARS", 2000),
        max_concurrency=_int_env("MAX_CONCURRENCY", 5),
        llm_timeout_seconds=_int_env("LLM_TIMEOUT_SECONDS", 60),
        max_retries=_int_env("MAX_RETRIES", 2, minimum=0),
        app_mode=_mode_env(),
        demo_dir=os.environ.get("REVIEWLENS_DEMO_DIR", "demo_data").strip() or "demo_data",
    )
