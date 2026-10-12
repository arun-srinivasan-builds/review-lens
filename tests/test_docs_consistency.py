"""Documentation must match the product: tab names, layout notes and the auth/hash procedure."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_readme_uses_the_current_tab_name() -> None:
    readme = _read("README.md")
    assert "Quality & Validation" not in readme  # renamed to Assignment Compliance
    assert "quality tab" not in readme.lower()
    assert "Assignment Compliance" in readme
    assert "assignment-compliance tab" in readme  # repository-layout line


def test_password_hash_procedure_names_traefik_not_caddy() -> None:
    for name in (".env.example", "docs/DEPLOY.md"):
        text = _read(name)
        assert "caddy hash-password" not in text  # the proxy is Traefik
        assert "Traefik" in text
        assert "htpasswd -nBC 12 admin" in text


def test_compose_really_uses_traefik_basic_auth() -> None:
    compose = _read("docker-compose.yml")
    assert "basicauth.users" in compose
    assert "ADMIN_PASSWORD_HASH" in compose
    assert "caddy" not in compose.lower().replace("# ", "")  # no leftover Caddy service
