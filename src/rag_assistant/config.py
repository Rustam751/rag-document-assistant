from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env."""

    anthropic_api_key: str | None = None  # falls back to the ANTHROPIC_API_KEY env var
    anthropic_model: str = "claude-opus-4-8"
    chroma_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"

    chunk_size: int = 1200  # characters per chunk
    chunk_overlap: int = 200  # characters of overlap between adjacent chunks
    top_k: int = 10  # retrieved chunks per question (eval showed k=5 misses supporting passages)
    max_answer_tokens: int = 16000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
