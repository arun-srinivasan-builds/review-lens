"""LLM provider factory. Only OpenAI (ChatOpenAI) is implemented initially."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from reviewlens.config import ConfigError, Settings


def get_llm(settings: Settings) -> BaseChatModel:
    """Return the configured chat model. Raises ConfigError (secret-safe) if misconfigured."""
    if settings.llm_provider != "openai":
        raise ConfigError(f"Unsupported LLM provider: {settings.llm_provider!r}. Only 'openai' is supported.")
    if not settings.openai_api_key.get_secret_value():
        raise ConfigError("OPENAI_API_KEY is not set. Add it to your .env file.")
    if not settings.openai_model:
        raise ConfigError("OPENAI_MODEL is not set. Add it to your .env file.")
    # Newer models reject non-default temperatures, so temperature is only sent when explicitly configured.
    extra: dict[str, float] = (
        {} if settings.openai_temperature is None else {"temperature": settings.openai_temperature}
    )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # retries are owned by reviewlens.classification, not the SDK
        **extra,
    )
