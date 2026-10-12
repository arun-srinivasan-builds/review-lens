"""Security tests: secret handling, logging hygiene, repo hygiene, upload limits. No network."""

from __future__ import annotations

import ast
import io
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import SecretStr

from reviewlens.classification import (
    build_review_chain,
    classify_reviews,
    retry_invalid_review,
)
from reviewlens.config import ConfigError, Settings, load_settings
from reviewlens.llm import get_llm
from reviewlens.models import REQUIRED_COLUMNS
from reviewlens.report import ReportGenerationError, generate_weekly_report
from reviewlens.security import safe_error_message
from reviewlens.validation import UploadError, load_reviews, validate_reviews
from tests.fakes import (
    FAKE_KEY,
    PLAIN_FAKE_KEY,
    FakeChatLLM,
    deterministic_analysis,
    fake_report,
)

ROOT = Path(__file__).resolve().parent.parent
KEYS = [FAKE_KEY, PLAIN_FAKE_KEY]


class AuthenticationError(Exception):
    status_code = 401


class RateLimitError(Exception):
    status_code = 429


def _settings(key: str, **kw: Any) -> Settings:
    return Settings(openai_api_key=SecretStr(key), openai_model="fake", max_retries=2, max_concurrency=2, **kw)


def _df(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"R{i}" for i in range(len(texts))],
            "date": ["2025-01-01"] * len(texts),
            "platform": ["Zomato"] * len(texts),
            "rating": [3.0] * len(texts),
            "review_text": texts,
        },
        columns=list(REQUIRED_COLUMNS),
    )


class _Upload(io.BytesIO):
    def __init__(self, data: bytes, name: str = "r.csv") -> None:
        super().__init__(data)
        self.name = name


# ---------------------------------------------------------------- secrets in Settings


@pytest.mark.parametrize("key", KEYS)
def test_settings_str_repr_do_not_expose_key(key: str) -> None:
    s = _settings(key)
    for text in (str(s), repr(s), f"{s!r}", f"{s}", str(s.openai_api_key), repr(s.openai_api_key)):
        assert key not in text
    assert s.openai_api_key.get_secret_value() == key


def test_load_settings_reads_env_but_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PLAIN_FAKE_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "fake")
    s = load_settings()
    assert PLAIN_FAKE_KEY not in repr(s) and PLAIN_FAKE_KEY not in str(s)


def test_chat_model_repr_hides_key() -> None:
    llm = get_llm(_settings(FAKE_KEY))
    assert FAKE_KEY not in repr(llm) and FAKE_KEY not in str(llm)


def test_config_errors_are_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("MAX_ROWS", "not-a-number")
    with pytest.raises(ConfigError) as ei:
        load_settings()
    assert FAKE_KEY not in str(ei.value)
    with pytest.raises(ConfigError) as ei2:
        get_llm(Settings(llm_provider="anthropic", openai_api_key=SecretStr(FAKE_KEY)))
    assert FAKE_KEY not in str(ei2.value)
    with pytest.raises(ConfigError) as ei3:
        get_llm(Settings(openai_api_key=SecretStr("")))
    assert "OPENAI_API_KEY" in str(ei3.value)  # names the variable, never a value


# ---------------------------------------------------------------- secrets in logs / errors


@pytest.mark.parametrize("key", KEYS)
def test_no_key_in_logs_on_auth_failure(key: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    def boom(text: str) -> Any:
        raise AuthenticationError(f"Incorrect API key provided: {key}. Check your key.")

    cfg = _settings(key)
    items = classify_reviews(_df(["a", "b", "c"]), cfg, llm=FakeChatLLM(review_handler=boom))  # type: ignore[arg-type]
    assert [i.processing_status for i in items] == ["needs_review"] * 3
    assert key not in caplog.text
    assert key not in "".join(i.model_dump_json() for i in items)


@pytest.mark.parametrize("key", KEYS)
def test_no_key_in_logs_on_transient_and_retry_paths(key: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    def boom(text: str) -> Any:
        raise RateLimitError(f"429 for key {key}")

    cfg = _settings(key)
    classify_reviews(_df(["a"]), cfg, llm=FakeChatLLM(review_handler=boom))  # type: ignore[arg-type]
    chain = build_review_chain(llm=FakeChatLLM(review_handler=boom))  # type: ignore[arg-type]
    retry_invalid_review(_df(["a"]).iloc[0], chain, cfg)
    assert key not in caplog.text


@pytest.mark.parametrize("key", KEYS)
def test_report_error_message_is_secret_free(key: str, caplog: pytest.LogCaptureFixture) -> None:
    from reviewlens.aggregation import calculate_weekly_metrics
    from reviewlens.classification import enriched_to_dataframe

    caplog.set_level(logging.DEBUG)
    cfg = _settings(key)
    items = classify_reviews(_df(["x", "y"]), cfg, llm=FakeChatLLM())  # type: ignore[arg-type]
    metrics = calculate_weekly_metrics(enriched_to_dataframe(items))

    def auth() -> Any:
        raise AuthenticationError(f"bad key {key}")

    def generic() -> Any:
        raise RuntimeError(f"upstream said: {key}")

    for handler in (auth, generic):
        with pytest.raises(ReportGenerationError) as ei:
            generate_weekly_report(metrics, ["e"], llm=FakeChatLLM(report_handler=handler), settings=cfg)  # type: ignore[arg-type]
        assert key not in str(ei.value)
        assert key not in repr(ei.value)
    assert key not in caplog.text


def test_safe_error_message_redacts_and_truncates() -> None:
    exc = RuntimeError(f"{FAKE_KEY} and {PLAIN_FAKE_KEY} " + "x" * 1000)
    msg = safe_error_message(exc, [PLAIN_FAKE_KEY])
    assert FAKE_KEY not in msg and PLAIN_FAKE_KEY not in msg
    assert len(msg) <= 300
    assert "Traceback" not in msg


def test_review_text_not_logged_on_failure_paths(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    secret_text = "UNIQUE-CUSTOMER-PRIVATE-TEXT-9f3a"

    def bad(text: str) -> Any:
        return {"sentiment": "nope"}

    classify_reviews(_df([secret_text]), _settings(FAKE_KEY), llm=FakeChatLLM(review_handler=bad))  # type: ignore[arg-type]
    classify_reviews(_df([secret_text]), _settings(FAKE_KEY), llm=FakeChatLLM(review_handler=deterministic_analysis))  # type: ignore[arg-type]
    validate_reviews(_df([secret_text, ""]), _settings(FAKE_KEY))
    assert secret_text not in caplog.text


def test_upload_errors_do_not_echo_file_content() -> None:
    cfg = _settings(FAKE_KEY)
    payload = b"review_id,date\nSECRET-CELL-VALUE,1\n"
    with pytest.raises(UploadError) as ei:
        load_reviews(_Upload(payload), cfg)
    assert "SECRET-CELL-VALUE" not in str(ei.value)


# ---------------------------------------------------------------- repository hygiene


def test_env_file_is_git_ignored() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert res.returncode == 0, ".env must be ignored by git"
    res2 = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert res2.returncode == 1, ".env.example must remain trackable"


def test_env_not_tracked_by_git() -> None:
    res = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert res.stdout.strip() == ""


FORBIDDEN_PACKAGES = ("langchain-google-genai", "langchain-anthropic", "anthropic", "google-generativeai")


@pytest.mark.parametrize("filename", ["requirements.txt", "requirements-dev.txt"])
def test_requirements_have_no_other_providers(filename: str) -> None:
    names = []
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.append(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    for pkg in FORBIDDEN_PACKAGES:
        assert pkg not in names, f"{pkg} must not be a dependency"


def test_requirements_pin_openai_integration() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "langchain-openai" in text and "streamlit" in text


def test_source_does_not_import_other_providers() -> None:
    banned = {"anthropic", "google.generativeai", "langchain_anthropic", "langchain_google_genai"}
    for path in [*(ROOT / "reviewlens").glob("*.py"), *(ROOT / "ui").glob("*.py"), ROOT / "app.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not any(m == b or m.startswith(b + ".") for b in banned), f"{path.name} imports {m}"


def test_source_has_no_print_or_hardcoded_keys() -> None:
    for path in [*(ROOT / "reviewlens").glob("*.py"), *(ROOT / "ui").glob("*.py"), ROOT / "app.py"]:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"print() in {path.name}:{node.lineno}"
        assert not re.search(r"sk-[A-Za-z0-9]{20,}", src), f"key-like literal in {path.name}"


def test_env_example_has_no_real_looking_key() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_\-]{16,}", text)
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", text)
    values = dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#")
    )
    assert "OPENAI_API_KEY" in values
    assert not re.fullmatch(r"[A-Za-z0-9_\-]{40,}", values["OPENAI_API_KEY"].strip())


# ---------------------------------------------------------------- upload limits via Settings


CSV_HEAD = "review_id,date,platform,rating,review_text\n"


def _csv(rows: int, text: str = "ok food") -> bytes:
    return (CSV_HEAD + "".join(f"R{i},2025-01-0{1 + i % 9},Zomato,4,{text}\n" for i in range(rows))).encode()


def test_oversize_upload_rejected_with_configured_limit() -> None:
    cfg = _settings(FAKE_KEY, max_upload_mb=1)
    big = _csv(1) + b"#" * (1024 * 1024 + 1)
    with pytest.raises(UploadError, match="too large|1 MB"):
        load_reviews(_Upload(big), cfg)
    assert len(load_reviews(_Upload(_csv(3)), cfg)) == 3


def test_row_cap_enforced() -> None:
    cfg = _settings(FAKE_KEY, max_rows=5)
    with pytest.raises(UploadError, match="Too many rows"):
        load_reviews(_Upload(_csv(6)), cfg)
    assert len(load_reviews(_Upload(_csv(5)), cfg)) == 5


def test_long_review_rejected_not_truncated_silently() -> None:
    cfg = _settings(FAKE_KEY, max_review_chars=50)
    df = load_reviews(_Upload(_csv(1, "x" * 51) + b"R9,2025-01-02,Zomato,4,short\n"), cfg)
    res = validate_reviews(df, cfg)
    assert res.valid_reviews["review_id"].tolist() == ["R9"]
    assert "longer than 50" in res.rejected_rows.iloc[0]["reject_reason"]


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("r.txt", _csv(1)),
        ("r.csv.exe", _csv(1)),
        ("r.csv", b"\xff\xfe\x00bad"),
        ("r.csv", b""),
        ("r.csv", CSV_HEAD.encode()),
        ("r.csv", b"a,b\n1,2\n"),
    ],
)
def test_bad_uploads_raise_user_friendly_error(name: str, data: bytes) -> None:
    with pytest.raises(UploadError):
        load_reviews(_Upload(data, name), _settings(FAKE_KEY))


def test_control_chars_and_nul_do_not_reach_prompt() -> None:
    cfg = _settings(FAKE_KEY)
    data = CSV_HEAD.encode() + b"R1,2025-01-01,Zomato,4,\"bad\x00\x07 text\"\n"
    res = validate_reviews(load_reviews(_Upload(data), cfg), cfg)
    assert res.valid_reviews.iloc[0]["review_text"] == "bad text"
    llm = FakeChatLLM()
    classify_reviews(res.valid_reviews, cfg, llm=llm)  # type: ignore[arg-type]
    assert "\x00" not in str(llm.review_prompts[0].to_messages()[-1].content)


def test_fake_report_helper_sanity() -> None:
    assert len(fake_report().action_items) == 3
