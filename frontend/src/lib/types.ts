import type {
  ClusterTarget,
  HostResources,
  ModelLaunchRequest,
  ModelPreferences,
  RuntimeHost,
  SetupJob,
  SetupCheck,
  HardwareTopology,
} from "./schema";
export type ClusterState = {
  clusters: ClusterTarget[];
  active_cluster_id: string | null;
};
export type Capacity = {
  cluster_id: string;
  cluster_name: string;
  available: boolean;
  fits: boolean;
  can_load?: boolean;
  required_gib?: number;
  available_gib?: number;
  missing_gib?: number;
  reason?: string | null;
};
export type Runtime = {
  engine_id: string;
  name: string;
  executable: boolean;
  reason: string;
  warnings: string[];
  blockers: string[];
};
export type Placement = {
  cluster_id: string;
  cluster_name: string;
  keep_loaded: boolean;
  gpu_ids?: string[];
  budget_gib?: Record<string, number>;
  concurrency?: { running: number; limit: number; waiting: number };
};
export type PoolModel = {
  id: string;
  size_bytes: number;
  placements: Placement[];
  cluster_capacity: Capacity[];
  loadable: boolean;
  editable_settings: boolean;
  preferences: ModelPreferences;
  last_settings: ModelLaunchRequest | null;
  runtime: Runtime | null;
  runtime_models?: { data?: { task?: string }[] };
};
export type Resident = { model_id: string; gpu_ids?: string[] };
export type PoolCluster = {
  id: string;
  name: string;
  available: boolean;
  age_seconds: number;
  busy: boolean;
  active_model?: string;
  resident_models?: Resident[];
  machines: {
    available?: boolean;
    memory_used_gib?: number;
    memory_total_gib?: number;
    host?: HostResources;
    accelerators?: number;
  }[];
  gpus?: {
    id: string;
    index: number;
    total_gib: number;
    reserved_gib: number;
    external_gib: number;
    used_gib: number;
    available_gib: number;
  }[];
};
export type PoolJob = {
  id: string;
  model_id: string;
  status: string;
  progress: number;
  message: string;
  runtime?: Runtime;
};
export type PoolState = {
  models: PoolModel[];
  clusters: PoolCluster[];
  jobs: PoolJob[];
};
export type RequestRecord = {
  id: string;
  model_id: string;
  cluster_id: string | null;
  started_at: number;
  outcome?: string;
  state?: string;
  duration_ms?: number;
  elapsed_ms?: number;
  input_tokens: number | null;
  output_tokens: number | null;
  memory_end?: { used_gib: number; total_gib: number };
  memory_start?: { used_gib: number; total_gib: number };
};
export type Activity = {
  since: number;
  totals: Record<string, number>;
  runtime_metrics: {
    available: boolean;
    sampled_at: string;
    decode_tokens_per_second: number | null;
  };
  pools: PoolCluster[];
  active_requests: number;
  queued_requests: number;
  latency: Record<string, number | null>;
  active: RequestRecord[];
  recent: RequestRecord[];
  retained_models: string[];
  history: { timestamp: number; output_tokens: number }[];
  groups: {
    model_id: string;
    cluster_id: string;
    requests: number;
    input_tokens: number;
    output_tokens: number;
  }[];
  coverage: string;
  persistence: { healthy: boolean; pending: number; dropped: number };
};
export type Performance = {
  summary: Record<string, number | null>;
  entities: { scope: string; entity: string }[];
  history: (Record<string, number | null> & {
    timestamp: number;
    resolution_seconds: number;
  })[];
  coverage: string;
  end: number;
  retention: {
    minute_days: number;
    hour_days: number;
    day_days: number;
    max_rows: number;
  };
  persistence: { healthy: boolean; pending: number; dropped: number };
};
export type Candidate = {
  address: string;
  hostname: string;
  source?: string;
  username?: string;
  fingerprint?: string;
  role?: "head" | "worker";
  fabric_address?: string | null;
  bootstrap_id?: string;
  access_command?: string;
  access_status?: string;
  access_error?: string;
  machine_kind?: string;
  setup_mode?: string;
  setup_supported?: boolean;
  install_supported?: boolean;
  platform?: string;
  architecture?: string;
  accelerators?: string[];
  memory_gib?: number;
};
export type Enrollment = {
  id: string;
  status: string;
  command?: string;
  error?: string;
  username?: string;
  fingerprint?: string;
};
export type SetupState = {
  enabled: boolean;
  mode: string;
  jobs: SetupJob[];
  connected_cluster?: { cluster_name: string };
};
export type SetupReport = {
  nodes: Candidate[];
  checks: SetupCheck[];
  topology: HardwareTopology;
};
export type PackageRuntime = {
  id: string;
  name: string;
  description: string;
  platforms: string[];
  launch_adapter: string | null;
  native_repository?: string;
  project_url: string;
  hardware: string;
  note: string;
};
export type Artifact = {
  id: string;
  runtime_id: string;
  version: string;
  platform: string;
  archive_bytes: number;
  artifact_kind: string;
};
export type PackageJob = {
  id: string;
  operation: string;
  status: string;
  progress: number;
  message: string;
  cluster_name?: string;
  runtime_id?: string;
  request: { runtime_id?: string; version?: string; artifact_id?: string };
};
export type Library = {
  runtimes: PackageRuntime[];
  artifacts: Artifact[];
  jobs: PackageJob[];
  hosts: RuntimeHost[];
  download_available: boolean;
  message: string;
};
