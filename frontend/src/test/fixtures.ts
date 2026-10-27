import type {
  ClusterSnapshot,
  ClusterTarget,
  HostResources,
  CatalogDetail,
  ModelFitPlan,
  RuntimeGuide,
} from "@/lib/schema";
import type { PoolModel, PoolState, Activity, Library } from "@/lib/types";
import rawSnapshot from "./snapshot.json";
import rawCatalog from "./catalog.json";
import rawFit from "./fit.json";
import telemetry from "./telemetry.json";
import { defaultLaunch } from "@/lib/launch-defaults";
export const host: HostResources = {
  id: "host-one",
  hostname: "Test server",
  ram_total_gib: 64,
  ram_used_gib: 20,
  ram_available_gib: 44,
  cpu_physical_cores: 4,
  cpu_logical_cores: 8,
  cpu_utilization: 25,
  cpu_cores: [
    { id: 0, utilization: 40 },
    { id: 1, utilization: 10 },
  ],
  last_seen: new Date().toISOString(),
};
export const snapshot: ClusterSnapshot = {
  ...rawSnapshot,
  timestamp: new Date().toISOString(),
  nodes: rawSnapshot.nodes.map((n) => ({ ...n, host })),
} as ClusterSnapshot;
export const cluster: ClusterTarget = {
  id: "pool-one",
  name: "Test pool",
  kind: "local-docker",
  hardware: "nvidia-gpu",
  status: "ready",
  endpoint: null,
  host_address: "192.168.50.10",
  runtime_port: 30000,
  gpu_count: 2,
  gpu_memory_gib: 24,
  gpu_model: "Test GPU",
  gpus: [
    {
      index: 0,
      name: "Test GPU",
      uuid: "gpu-1",
      memory_gib: 24,
      compute_capability: "8.6",
    },
  ],
  model_storage_path: null,
  active: true,
  created_at: new Date().toISOString(),
  capabilities: [],
};
export const model: PoolModel = {
  id: "acme/chat",
  size_bytes: 2 ** 30,
  placements: [],
  cluster_capacity: [
    {
      cluster_id: "pool-one",
      cluster_name: "Test pool",
      available: true,
      fits: true,
      can_load: true,
      required_gib: 3,
      available_gib: 20,
    },
  ],
  loadable: true,
  editable_settings: true,
  preferences: { keep_loaded: false, cluster_id: null, last_settings: null },
  last_settings: {
    ...defaultLaunch,
    model_id: "acme/chat",
    served_model_name: "chat",
    context_length: 8192,
  },
  runtime: {
    engine_id: "vllm",
    name: "vLLM",
    executable: true,
    reason: "Compatible transformer",
    warnings: [],
    blockers: [],
  },
};
export const pool: PoolState = {
  models: [model],
  clusters: [
    {
      id: cluster.id,
      name: cluster.name,
      available: true,
      age_seconds: 1,
      busy: false,
      resident_models: [],
      machines: [
        { available: true, host, memory_used_gib: 5, memory_total_gib: 24 },
      ],
      gpus: [
        {
          id: "gpu-1",
          index: 0,
          total_gib: 24,
          used_gib: 4,
          reserved_gib: 5,
          external_gib: 1,
          available_gib: 18,
        },
      ],
    },
  ],
  jobs: [],
};
export const catalog = rawCatalog as CatalogDetail;
export const fit = rawFit as ModelFitPlan;
export const guide: RuntimeGuide = {
  cluster_id: cluster.id,
  cluster_name: cluster.name,
  engine_id: "vllm",
  engine_name: "vLLM",
  model_id: null,
  compatible: true,
  prepared: false,
  update_available: false,
  platform: "linux/amd64",
  version: "1.0",
  image: "image:v1",
  digest: "sha256:test",
  size_bytes: 10,
  summary: "Install the pinned runtime on this compute pool.",
  blockers: [],
  alternatives: [],
  steps: [
    {
      id: "resolve",
      title: "Resolve runtime",
      description: "Verify immutable image",
    },
    {
      id: "install",
      title: "Install runtime",
      description: "Install on targets",
    },
  ],
};
export const activity: Activity = {
  since: 1700000000,
  totals: {
    requests: 3,
    successful: 2,
    failed: 1,
    cancelled: 0,
    input_tokens: 100,
    output_tokens: 20,
    cached_tokens: 4,
    unknown_usage: 0,
  },
  runtime_metrics: {
    available: false,
    sampled_at: new Date().toISOString(),
    decode_tokens_per_second: null,
  },
  pools: pool.clusters,
  active_requests: 0,
  queued_requests: 0,
  latency: {
    p50_ms: 10,
    p95_ms: 20,
    ttft_p50_ms: 5,
    ttft_p95_ms: 7,
    sample_size: 3,
  },
  active: [],
  recent: [
    {
      id: "req-1",
      model_id: "acme/chat",
      cluster_id: "pool-one",
      started_at: 1700000000,
      outcome: "success",
      duration_ms: 10,
      input_tokens: 100,
      output_tokens: 20,
    },
  ],
  retained_models: [],
  history: [],
  groups: [],
  coverage: "Captured requests only.",
  persistence: { healthy: true, pending: 0, dropped: 0 },
};
export const library: Library = {
  runtimes: [
    {
      id: "llama-cpp",
      name: "llama.cpp",
      description: "Native GGUF runtime",
      platforms: ["darwin/arm64", "linux/amd64"],
      launch_adapter: "llama-cpp",
      native_repository: "ggml-org/llama.cpp",
      project_url: "https://github.com/ggml-org/llama.cpp",
      hardware: "CPU / Metal / CUDA",
      note: "Checksum verified",
    },
  ],
  artifacts: [],
  jobs: [],
  hosts: [],
  download_available: true,
  message: "Verified packages for managed targets.",
};
export function baseline(path: string): unknown {
  const url = new URL(path, "http://localhost");
  switch (url.pathname) {
    case "/api/v1/clusters":
      return { clusters: [cluster], active_cluster_id: cluster.id };
    case "/api/v1/pool":
      return pool;
    case "/api/v1/snapshot":
      return snapshot;
    case "/api/v1/inference/activity":
      return activity;
    case "/api/v1/history":
      return [snapshot];
    case "/api/v1/downloads":
      return [];
    case "/api/v1/models":
      return [
        {
          id: model.id,
          revision: "main",
          source: "managed",
          path: null,
          size_bytes: 2 ** 30,
          downloaded_at: null,
          loaded: false,
          launch_supported: true,
        },
      ];
    case "/api/v1/catalog/models":
      return { models: [catalog], cursor: null };
    case "/api/v1/catalog/models/acme/chat":
      return catalog;
    case "/api/v1/models/fit":
    case "/api/v1/models/fit-local":
      return fit;
    case "/api/v1/controller/huggingface":
      return { configured: false, requires_admin: true };
    case "/api/v1/launches":
      return [];
    case "/api/v1/telemetry":
      return telemetry;
    case "/api/v1/optimization/benchmarks":
    case "/api/v1/cache/prefixes":
    case "/api/v1/logs":
    case "/api/v1/logs/instances":
      return [];
    case "/api/v1/setup":
      return { enabled: true, mode: "demo", jobs: [] };
    case "/api/v1/runtime-library":
      return library;
    case "/api/v1/runtime-management/engines":
      return { cluster_id: cluster.id, platform: "linux/amd64", engines: [] };
    case "/api/v1/runtime-management/guide":
      return guide;
    case "/api/v1/inference/performance":
      return {
        summary: { requests: 0 },
        entities: [],
        history: [],
        coverage: "Recorded data",
        end: 1700000000,
        retention: {
          minute_days: 7,
          hour_days: 90,
          day_days: 365,
          max_rows: 10000,
        },
        persistence: { healthy: true, pending: 0, dropped: 0 },
      };
    default:
      if (url.pathname.includes("/versions"))
        return {
          versions: [{ version: "b123", platforms: ["darwin/arm64"] }],
          next_cursor: null,
          ordering: "latest first",
          stale: false,
        };
      return {};
  }
}
