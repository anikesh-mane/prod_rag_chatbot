"""Application settings using Pydantic Settings for configuration management."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: SecretStr = Field(..., description="PostgreSQL connection string")
    pool_size: int = Field(default=5, ge=1, le=50)
    max_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = Field(default=False, description="Echo SQL queries")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=10, ge=1, le=100)
    ttl_default: int = Field(default=3600, description="Default TTL in seconds")


class MilvusSettings(BaseSettings):
    """Milvus vector database settings."""

    model_config = SettingsConfigDict(env_prefix="MILVUS_")

    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection_name: str = Field(default="rag_documents")
    index_type: str = Field(default="IVF_FLAT")
    metric_type: str = Field(default="COSINE")
    nlist: int = Field(default=128, description="Number of clusters for IVF index")
    nprobe: int = Field(default=16, description="Number of clusters to search")


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    provider: Literal["openai", "gemini", "mistral"] = Field(default="openai")
    model: str = Field(default="gpt-4")
    fallback_model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10)


class EmbeddingSettings(BaseSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["openai", "sentence-transformers"] = Field(default="openai")
    model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=1536)
    batch_size: int = Field(default=100, ge=1, le=1000)


class AuthSettings(BaseSettings):
    """Authentication settings."""

    model_config = SettingsConfigDict(env_prefix="AUTH_")

    jwt_secret_key: SecretStr = Field(..., description="Secret key for JWT signing")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=15, ge=1, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)


class RateLimitSettings(BaseSettings):
    """Rate limiting settings."""

    model_config = SettingsConfigDict(env_prefix="RATELIMIT_")

    chat_per_minute: int = Field(default=20, ge=1, le=1000)
    chat_per_hour: int = Field(default=500, ge=1, le=10000)
    feedback_per_hour: int = Field(default=100, ge=1, le=1000)
    api_key_per_hour: int = Field(default=1000, ge=1, le=100000)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline settings."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")


class IngestionSettings(BaseSettings):
    """Document ingestion settings."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size: int = Field(default=512, ge=100, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    max_document_size_mb: int = Field(default=10, ge=1, le=100)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # API
    api_title: str = Field(default="RAG Chatbot API")
    api_version: str = Field(default="1.0.0")
    api_prefix: str = Field(default="/api/v1")

    # Secrets
    openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    gemini_api_key: SecretStr = Field(default=SecretStr(""), description="Google Gemini API key")

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        return v.lower()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()
