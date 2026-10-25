from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPARK_CLUSTER_", env_file=".env")

    title: str = "Spark Cluster"
    mode: str = "demo"
    database_path: Path = Path("./spark-cluster.db")
    catalog_cache_path: Path | None = None
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
    runtime_node_port: int = 30000
    model_cache_path: Path = Path("./models")
    hf_token: str | None = None
    model_admin_token: str | None = None
    hf_download_workers: int = 4
    model_usable_memory_fraction: float = 0.82
    fallback_node_count: int = 1
    fallback_node_memory_gib: float = 128
    runtime_logs_mode: str = "auto"
    runtime_control_mode: str = "auto"
    runtime_log_poll_seconds: float = 2
    runtime_log_retention: int = 10_000
    kubernetes_namespace: str = "inference"
    kubernetes_api_url: str | None = None
    kubernetes_token_path: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    kubernetes_ca_path: Path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    runtime_pod_selector: str = "app.kubernetes.io/name=spark-cluster-runtime"
    runtime_head_deployment: str = "spark-cluster-runtime-head"
    runtime_worker_statefulset: str = "spark-cluster-runtime-worker"
    runtime_vllm_image: str = "nvcr.io/nvidia/vllm:25.09-py3"
    runtime_vllm_speculative_methods: str = "off,auto,mtp"
    runtime_sglang_image: str | None = None
    runtime_tensorfold_image: str | None = None
    runtime_deepinfer_image: str | None = None
    runtime_llama_cpp_image: str | None = None
    runtime_ollama_image: str | None = None
    runtime_tensorrt_llm_image: str | None = None
    runtime_triton_image: str | None = None
    runtime_tgi_image: str | None = None
    runtime_lmdeploy_image: str | None = None
    runtime_xinference_image: str | None = None
    runtime_localai_image: str | None = None
    runtime_cache_path: Path | None = None
    runtime_metrics_url: str | None = None
    runtime_metrics_poll_seconds: float = 5
    inference_retention_requests: int = Field(default=50_000, ge=100, le=1_000_000)
    inference_retention_days: int = Field(default=7, ge=1, le=30)
    performance_minute_days: int = Field(default=1, ge=1, le=7)
    performance_hour_days: int = Field(default=30, ge=7, le=90)
    performance_day_days: int = Field(default=180, ge=90, le=730)
    performance_max_rows: int = Field(default=150_000, ge=1000, le=1_000_000)
    setup_enabled: bool = False
    setup_state_path: Path = Path("./setup-state")
    setup_ssh_identity: Path | None = None
    setup_ssh_config: Path | None = None
    setup_image_repository: str = "ghcr.io/thibautrey/spark-cluster"
    setup_image_tag: str = "latest"
    setup_image_digest: str | None = None
    setup_registry_endpoint: str | None = None
    remote_cluster_url: str | None = None
    controller_public_url: str | None = None
    docker_runtime_enabled: bool = False
    docker_socket_path: Path = Path("/var/run/docker.sock")
    docker_host_model_path: Path = Path("/models")
    docker_runtime_host: str = "127.0.0.1"
    docker_runtime_port_start: int = 30000
    on_demand_profiles_path: Path | None = None
    active_hardware: str = "dgx-spark"
    active_compute_capability: float | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def direct_runtime_base_url(self) -> str:
        if self.remote_cluster_url:
            parsed = urlparse(self.remote_cluster_url)
            return f"{parsed.scheme}://{parsed.hostname}:{self.runtime_node_port}/v1"
        return self.openai_api_base.rstrip("/")
