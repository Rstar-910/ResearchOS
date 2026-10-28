from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "sqlite:///./researchos.db"
    discovery_enabled: bool = True
    searxng_url: HttpUrl | None = None
    semantic_scholar_api_key: SecretStr | None = None
    search_timeout: float = Field(default=15, ge=1, le=60)
    # A key alone selects Ollama Cloud; a base URL alone selects a local Ollama server.
    ollama_api_key: SecretStr | None = None
    ollama_base_url: HttpUrl | None = None
    llm_model: str = "gpt-oss:120b"
    llm_timeout: float = Field(default=120, ge=5, le=600)
    # Embeddings come from a local Ollama server; unset means keyword-only search.
    embedding_base_url: HttpUrl | None = None
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_fallback_model: str = "nomic-embed-text"

    @field_validator(
        "searxng_url",
        "semantic_scholar_api_key",
        "ollama_api_key",
        "ollama_base_url",
        "embedding_base_url",
        mode="before",
    )
    @classmethod
    def blank_optional_setting(cls, value):
        if isinstance(value, str):
            return value.strip() or None
        return value
