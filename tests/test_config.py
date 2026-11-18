import pytest

from app.core.config import Settings


def test_defaults_are_free_local_stack() -> None:
    s = Settings(_env_file=None)
    assert s.is_sqlite
    assert s.llm_provider == "ollama"
    assert s.max_upload_mb == 20


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/askdata")
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-2.0-flash"
    assert not s.is_sqlite


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require",
        "postgres://u:p@ep-x.neon.tech/neondb?sslmode=require",
    ],
)
def test_plain_postgres_urls_use_psycopg3(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setenv("DATABASE_URL", url)
    s = Settings(_env_file=None)
    assert s.database_url == "postgresql+psycopg://u:p@ep-x.neon.tech/neondb?sslmode=require"


def test_prod_requires_real_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(_env_file=None)
    monkeypatch.setenv("SECRET_KEY", "short")
    with pytest.raises(ValueError, match="SECRET_KEY"):
        Settings(_env_file=None)
    monkeypatch.setenv("SECRET_KEY", "x" * 40)
    assert Settings(_env_file=None).app_env == "prod"


def test_invalid_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="llm_provider"):
        Settings(_env_file=None)


def test_every_setting_is_documented_in_env_example() -> None:
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".env.example"
    documented = {
        line.split("=", 1)[0].strip()
        for line in example.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    internal = {"APP_NAME"}  # not meant to be configured
    missing = {name.upper() for name in Settings.model_fields} - documented - internal
    assert not missing, f"add to .env.example: {sorted(missing)}"
