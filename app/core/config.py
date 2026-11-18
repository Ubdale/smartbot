"""Application settings, loaded from environment variables and an optional `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["ollama", "gemini", "groq"]
DEFAULT_SECRET = "change-me-in-production"  # noqa: S105 - placeholder, rejected in prod


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AskData"
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # App database. SQLite by default; set a postgresql+psycopg:// URL to switch.
    database_url: str = "sqlite:///./data/askdata.db"

    # Security
    secret_key: SecretStr = SecretStr(DEFAULT_SECRET)
    access_token_expire_minutes: int = 60 * 24

    # LLM
    llm_provider: LLMProviderName = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Datasets
    data_dir: Path = Path("./data")
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    @model_validator(mode="after")
    def _require_real_secret_in_prod(self) -> Self:
        secret = self.secret_key.get_secret_value()
        if self.app_env == "prod" and (secret == DEFAULT_SECRET or len(secret) < 32):
            raise ValueError(
                "SECRET_KEY must be set to a random value of at least 32 characters in prod"
            )
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
