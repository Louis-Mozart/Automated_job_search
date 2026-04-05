from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings – loaded from environment variables or .env file."""

    # ── OpenAI ──────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # ── RapidAPI / JSearch ──────────────────────────────────────
    rapidapi_key: Optional[str] = Field(default=None)

    # ── Adzuna ──────────────────────────────────────────────────
    adzuna_app_id: Optional[str] = Field(default=None)
    adzuna_app_key: Optional[str] = Field(default=None)
    adzuna_country: str = Field(default="us")

    # ── App defaults ────────────────────────────────────────────
    default_k: int = Field(default=10)
    max_jobs_to_fetch: int = Field(default=600)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
