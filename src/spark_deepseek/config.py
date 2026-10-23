from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPARK_DEEPSEEK_", env_file=".env")

    title: str = "Spark DeepSeek"
    mode: str = "demo"
    database_path: Path = Path("./spark-deepseek.db")
    sample_interval_seconds: float = 2.0
    history_retention_hours: int = 24
    ingest_token: str | None = None
    cluster_name: str = "gb10-lab"
    prometheus_url: str | None = None
    cors_origins: str = "http://localhost:8080,http://localhost:5173"
    runtime_model: str = "deepseek-ai/DeepSeek-R1"
    runtime_quantization: str = "NVFP4"
    runtime_context_length: int = 524288
    runtime_tensor_parallel: int = 4
    runtime_pipeline_parallel: int = 1
    openai_api_base: str = "http://localhost:8000/v1"
    openai_api_key: str | None = None
    model_cache_path: Path = Path("./models")
    hf_token: str | None = None
    model_admin_token: str | None = None
    hf_download_workers: int = 4
    model_usable_memory_fraction: float = 0.82
    fallback_node_count: int = 1
    fallback_node_memory_gib: float = 128

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]
