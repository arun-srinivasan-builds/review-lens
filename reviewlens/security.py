"""Input sanitisation, secret redaction and safe error mapping."""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*]{4,}")


def sanitize_text(value: str, max_chars: int | None = None) -> str:
    """Remove control characters and trim. Keeps newlines/tabs; does not alter meaning."""
    cleaned = _CONTROL_CHARS.sub("", value).strip()
    if max_chars is not None and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def redact_secrets(message: str, secrets: list[str] | None = None) -> str:
    """Mask known secrets and anything that looks like an API key."""
    redacted = _KEY_PATTERN.sub("[REDACTED]", message)
    for secret in secrets or []:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def safe_error_message(exc: BaseException, secrets: list[str] | None = None) -> str:
    """User-facing error text: exception class + redacted message, no traceback."""
    return redact_secrets(f"{type(exc).__name__}: {exc}", secrets)[:300]


_ANGLE_TRANSLATION = str.maketrans({"<": "＜", ">": "＞"})


def escape_delimiters(value: str) -> str:
    """Replace angle brackets so untrusted text cannot close prompt delimiters like </review>."""
    return value.translate(_ANGLE_TRANSLATION)


def one_line(value: str, max_chars: int) -> str:
    """Collapse whitespace/newlines, neutralise delimiters and cap length for single-line prompt fields."""
    return escape_delimiters(" ".join(sanitize_text(value).split()))[:max_chars]
