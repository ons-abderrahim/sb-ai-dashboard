"""
Application Configuration
=========================
All settings are loaded from environment variables (or .env file).
Uses Pydantic Settings for type-safe config management.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ───────────────────────────────────────────────────────────────────
    app_version: str = "0.1.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    secret_key: str = Field(default="dev-secret", alias="SECRET_KEY")
    allowed_origins: List[str] = ["http://localhost:8050", "http://localhost:3000"]

    # ── InfluxDB ──────────────────────────────────────────────────────────────
    influxdb_url: str = Field(default="http://localhost:8086", alias="INFLUXDB_URL")
    influxdb_token: str = Field(default="dev-token", alias="INFLUXDB_TOKEN")
    influxdb_org: str = Field(default="smartbuilding", alias="INFLUXDB_ORG")
    influxdb_bucket: str = Field(default="sensors", alias="INFLUXDB_BUCKET")

    # ── TimescaleDB ───────────────────────────────────────────────────────────
    timescale_url: str = Field(
        default="postgresql+asyncpg://sbuser:sbpassword@localhost:5432/smartbuilding",
        alias="TIMESCALE_URL",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    rag_pipeline_url: str = Field(default="http://localhost:8001", alias="RAG_PIPELINE_URL")
    rag_api_key: str = Field(default="", alias="RAG_API_KEY")

    # ── Model Paths ───────────────────────────────────────────────────────────
    model_artifacts_dir: str = Field(default="models/", alias="MODEL_ARTIFACTS_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
