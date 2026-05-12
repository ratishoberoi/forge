"""Centralized application settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FORGE_",
        extra="ignore",
    )

    env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model_alias: str = "deepseek-coder"
    model_trust_remote_code: bool = True
    model_dtype: str = "auto"
    model_max_model_len: int | None = 16384
    model_gpu_memory_utilization: float = Field(default=0.92, gt=0.0, le=1.0)
    model_tensor_parallel_size: int = Field(default=1, ge=1)
    model_max_num_seqs: int = Field(default=16, ge=1)
    model_enable_prefix_caching: bool = True
    model_download_dir: str | None = None
    model_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    model_top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    model_max_tokens: int = Field(default=1024, ge=1)
    model_enforce_eager: bool = False
    model_quantization: str | None = None

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_model_alias: str = "bge-small"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_max_length: int = Field(default=512, ge=32)
    embedding_cache_path: str = ".forge/cache/embeddings.sqlite3"

    vector_db_path: str = ".forge/qdrant"
    vector_collection: str = "forge_repo_chunks"

    repo_default_root: str | None = None
    repo_ignore_patterns: tuple[str, ...] = (
        ".git",
        "node_modules",
        ".venv",
        "dist",
        "build",
        "__pycache__",
        "*.pyc",
    )
    repo_respect_gitignore: bool = True
    repo_max_file_bytes: int = Field(default=524288, ge=1024)
    repo_index_state_path: str = ".forge/index_state.json"
    repo_incremental: bool = True
    repo_retrieval_limit: int = Field(default=8, ge=1)
    repo_graph_neighbors: int = Field(default=6, ge=1)
    repo_watcher_enabled: bool = False
    repo_watcher_debounce_ms: int = Field(default=750, ge=50)

    runtime_max_concurrency: int = Field(default=4, ge=1)
    runtime_default_timeout_ms: int = Field(default=30000, ge=100)
    runtime_max_retries: int = Field(default=2, ge=0)
    runtime_event_queue_size: int = Field(default=1024, ge=16)
    runtime_context_token_budget: int = Field(default=6000, ge=256)
    runtime_agent_heartbeat_ms: int = Field(default=250, ge=10)

    @field_validator("repo_ignore_patterns", mode="before")
    @classmethod
    def parse_repo_ignore_patterns(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(part).strip() for part in value if str(part).strip())
        raise TypeError("repo_ignore_patterns must be a comma-separated string or sequence.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
