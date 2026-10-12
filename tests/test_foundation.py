from dataclasses import replace

import pytest
from pydantic import SecretStr, ValidationError

from reviewlens.config import ConfigError, Settings
from reviewlens.llm import get_llm
from reviewlens.models import ReviewAnalysis, WeeklyReport
from reviewlens.security import redact_secrets, safe_error_message

FAKE_KEY = "sk-test1234567890abcdef"


def _settings(**overrides: object) -> Settings:
    base = Settings(openai_api_key=SecretStr(FAKE_KEY), openai_model="test-model")
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_get_llm_returns_chat_openai() -> None:
    llm = get_llm(_settings())
    assert type(llm).__name__ == "ChatOpenAI"


def test_missing_key_raises_secret_safe_error() -> None:
    with pytest.raises(ConfigError) as exc:
        get_llm(_settings(openai_api_key=SecretStr("")))
    assert "OPENAI_API_KEY" in str(exc.value)


def test_missing_model_raises() -> None:
    with pytest.raises(ConfigError):
        get_llm(_settings(openai_model=""))


def test_unsupported_provider_raises() -> None:
    with pytest.raises(ConfigError):
        get_llm(_settings(llm_provider="gemini"))


def test_settings_repr_hides_key() -> None:
    assert FAKE_KEY not in repr(_settings())


def test_redaction_masks_keys() -> None:
    assert FAKE_KEY not in redact_secrets(f"failed with {FAKE_KEY}")
    assert FAKE_KEY not in safe_error_message(RuntimeError(f"bad key {FAKE_KEY}"))


def test_analysis_enums_enforced() -> None:
    with pytest.raises(ValidationError):
        ReviewAnalysis(
            sentiment="happy",  # type: ignore[arg-type]
            category="delivery",
            urgency="low",
            language="English",
            summary="x",
            suggested_reply="y",
        )


def test_report_requires_exactly_three_actions() -> None:
    with pytest.raises(ValidationError):
        WeeklyReport(top_3_complaints=[], top_praises=[], urgent_attention="", action_items=["a", "b"])
