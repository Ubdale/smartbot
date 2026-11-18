"""Application settings, loaded from environment variables and an optional `.env` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["ollama", "gemini", "groq"]
StorageBackend = Literal["local", "s3"]

# Used when LLM_MODEL is empty, so switching LLM_PROVIDER alone is enough.
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen2.5:7b-instruct",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
}
# Free tiers limit requests per minute *per model*, so summaries use a second model
# (when the provider has a good small one) to roughly double free throughput.
DEFAULT_SUMMARY_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "openai/gpt-oss-20b",
}
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
    llm_model: str | None = None  # empty = the provider's default (DEFAULT_MODELS)
    llm_summary_model: str | None = None  # empty = DEFAULT_SUMMARY_MODELS, else llm_model
    # Used when the main provider fails or hits its free-tier limit (e.g. gemini -> groq).
    llm_fallback_provider: LLMProviderName | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Layers to put on the GPU. 0 = CPU only (use on laptops with small GPUs, which
    # otherwise crash with "CUDA out of memory"). Unset = let Ollama decide.
    ollama_num_gpu: int | None = None
    ollama_timeout_seconds: float = Field(default=300, gt=0)
    # Bearer token for a self-hosted Ollama behind a reverse proxy (deploy/oracle). Ollama
    # itself has no authentication, so never expose it to the internet without one.
    ollama_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Datasets. "local" keeps files on disk; "s3" also stores them in S3-compatible
    # object storage (e.g. Neon) so they survive restarts on hosts with temporary disks.
    data_dir: Path = Path("./data")
    max_upload_mb: int = Field(default=20, ge=1, le=200)
    storage_backend: StorageBackend = "local"
    s3_endpoint_url: str | None = Field(
        default=None, validation_alias=AliasChoices("S3_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3")
    )
    s3_bucket: str | None = None
    s3_prefix: str = ""  # optional key prefix, e.g. "staging/"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    aws_region: str = "us-east-1"

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3(cls, v: str) -> str:
        # Providers like Neon hand out plain postgres(ql):// URLs; SQLAlchemy would pick psycopg2.
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v.removeprefix(prefix)
        return v

    @model_validator(mode="after")
    def _require_real_secret_in_prod(self) -> Self:
        secret = self.secret_key.get_secret_value()
        if self.app_env == "prod" and (secret == DEFAULT_SECRET or len(secret) < 32):
            raise ValueError(
                "SECRET_KEY must be set to a random value of at least 32 characters in prod"
            )
        return self

    @model_validator(mode="after")
    def _require_credentials_for_chosen_services(self) -> Self:
        """Fail at startup, not on the first user request."""
        if self.storage_backend == "s3":
            missing = [
                name
                for name, value in [
                    ("S3_BUCKET", self.s3_bucket),
                    ("AWS_ACCESS_KEY_ID", self.aws_access_key_id),
                    ("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key),
                ]
                if not value
            ]
            if missing:
                raise ValueError(f"STORAGE_BACKEND=s3 needs: {', '.join(missing)}")
        if self.app_env == "prod":
            keys = {"gemini": self.gemini_api_key, "groq": self.groq_api_key}
            for setting, provider in [
                ("LLM_PROVIDER", self.llm_provider),
                ("LLM_FALLBACK_PROVIDER", self.llm_fallback_provider),
            ]:
                if provider in keys and not keys[provider]:
                    raise ValueError(f"{setting}={provider} needs its API key in prod")
        return self

    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    @property
    def summary_model_name(self) -> str:
        if self.llm_summary_model:
            return self.llm_summary_model
        if self.llm_model:  # a custom model was chosen: use it for both unless told otherwise
            return self.llm_model
        return DEFAULT_SUMMARY_MODELS.get(self.llm_provider, self.model_name)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()
