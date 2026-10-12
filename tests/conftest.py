"""Test isolation: no real .env, no real API key, no sleeping, no network."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from reviewlens import classification, report
from reviewlens.config import Settings
from tests.fakes import FAKE_KEY

_ENV_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LLM_PROVIDER",
    "MAX_UPLOAD_MB",
    "MAX_ROWS",
    "MAX_REVIEW_CHARS",
    "MAX_CONCURRENCY",
    "LLM_TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "REVIEWLENS_MODE",
    "REVIEWLENS_DEMO_DIR",
)


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's real .env/OPENAI_API_KEY must never reach (or leak from) a test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("reviewlens.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(classification, "_sleep", lambda s: None)
    monkeypatch.setattr(report, "_sleep", lambda s: None)


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr(FAKE_KEY),
        openai_model="fake-model",
        max_concurrency=4,
        max_retries=2,
    )
