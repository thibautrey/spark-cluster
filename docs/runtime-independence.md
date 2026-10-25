# Controller-independent inference

Spark Cluster separates management traffic from the inference data path. OpenAI-compatible clients
connect directly to the runtime service on the cluster head. The controller only validates a launch,
patches Kubernetes resources, and observes their state. It never proxies `/v1/models`,
`/v1/chat/completions`, streaming tokens, or embeddings.

## Independent lifecycle

Guided setup installs two Helm releases in the `inference` namespace:

- `spark-cluster-controller` owns the dashboard, bridge, agents, credentials, RBAC and controller data.
- `spark-cluster-runtime` owns the OpenAI service, inference head, Ray workers and LMCache config.

Updating, restarting, deleting or temporarily disconnecting the controller release does not scale,
restart or delete runtime workloads. The controller shutdown path only stops its local work queue; it
never patches runtime replicas. A runtime upgrade is a separate, explicit operation against the
`spark-cluster-runtime` release.

The default guided topology exposes:

- controller/bridge: `http://HEAD_LAN_IP:30080`
- direct OpenAI API: `http://HEAD_LAN_IP:30000/v1`

The controller advertises the second URL with `routing_mode=direct-runtime` and
`controller_required=false`. Requests sent to `/v1` on the controller receive HTTP 421 with the direct
runtime endpoint. This prevents an accidental controller dependency from silently reaching
production.

## Failure behavior

Once a runtime is ready, controller failure has no effect on existing inference requests or new
requests sent to port 30000. On restart, interrupted launch jobs are marked for review and are never
replayed automatically, because replaying a Kubernetes patch could roll a healthy runtime. The following
management functions are unavailable until the controller
returns: launching a different model, changing runtime parameters, dashboard telemetry, log browsing,
model downloads and assisted recovery. Kubernetes continues supervising the runtime independently.

A controller update should use only the controller release:

```bash
helm upgrade spark-cluster-controller ./spark-cluster \
  --namespace inference --reuse-values --set image.tag=NEW_CONTROLLER_TAG
```

Runtime changes use the other release explicitly:

```bash
helm upgrade spark-cluster-runtime ./spark-cluster \
  --namespace inference --reuse-values --set runtime.image.tag=NEW_RUNTIME_TAG
```

Do not combine the two commands in an automatic controller updater.

## Existing single-release installations

Older installations may have a single `spark-cluster` Helm release owning both workloads. Migrate
that release during a planned maintenance window before relying on independent upgrades. New guided
installations use split ownership automatically. Never import the same Kubernetes object into both
releases.

## Opt-in on-demand Docker inference

For a local GPU pool, set `SPARK_CLUSTER_ON_DEMAND_PROFILES_PATH` to a private (0600)
JSON recipe file. This explicitly selects a different lifecycle: the controller owns stable
HTTP listeners, creates temporary Docker runtimes when inference arrives, serializes requests
across the GPU pool, streams responses, and removes runtimes after `idle_seconds` (default 120).
An in-flight response holds the lease until its last byte. Switching models removes the previous
runtime first. Cold starts can take several minutes; callers must allow sufficient request timeouts.
Health and model-list probes do not load models or reset inactivity. A model-list result advertises
availability through the orchestrator, not residency in GPU memory.

Each recipe specifies `id`, `model`, `listen_host`, `listen_port`, `upstream_port`,
`readiness_path`, `idle_seconds`, `startup_seconds`, and a trusted Docker create specification
in `container`. Bind the runtime upstream to loopback on a different port. Keep credentials in
this private file or mounted secrets; never commit recipes containing deployment credentials.
The proxy preserves upstream authentication and does not log request bodies or credentials.
Recipe files can create Docker containers and must be managed only by trusted administrators.

For services that load weights lazily, set the optional `preload_path` to a runtime-relative
POST endpoint. For Speaches Whisper, use `/api/ps/deepdml/faster-whisper-large-v3-turbo-ct2`
with `WHISPER__TTL=-1`. The controller waits for this request to succeed within the startup
deadline before reporting the runtime ready; a healthy HTTP service alone is insufficient.
Use explicit `resources` to allow automatic GPU placement for an imported service. Without
that declaration, imported recipes preserve their original device affinity and require an
exclusive device when no engine memory budget is available.

Public listeners can preserve legacy service ports. Clients on private Docker DNS networks must
instead use a reachable host listener. `/api/v1/inference/status` reports profiles and residency
without exposing Docker recipes. Unrelated application containers are never stopped. Manual local
Docker launches are blocked while this mode owns the GPU pool. Disable legacy restart policies,
Unraid autostart, and external supervisors before enabling the corresponding listeners.

Unlike direct-runtime mode above, this mode depends on the controller for inference. Controller
shutdown drains requests and removes its temporary runtime; crash recovery reclaims only containers
labelled with the matching `spark-cluster.on-demand` profile. Docker runtimes have no restart policy.
The ordinary direct-runtime path and remote clusters are unchanged when no recipes are configured.

## Model pool and retention

The model pool is independent of the dashboard's selected cluster. `/api/v1/pool` merges
installed models and live residency across local Docker and authenticated remote bridges.
The model list provides **Load model**, **Settings**, and **Unload**. Load reuses saved launch
parameters, or the exact imported service recipe. A first generic launch uses conservative
8k context and one GPU, followed by a memory and engine preflight on each candidate.

**Keep loaded** protects a model from both idle unloading and eviction. This setting is
persistent. Retained Docker runtimes survive controller restarts and are adopted again only
when their ownership labels match. Other loads use a different compatible cluster or wait
with a visible reason. **Unload** explicitly releases the model and clears retention.
A saved cluster is a strict requirement for subsequent loads; **Automatic — pool** prefers
existing residency, then idle clusters, then eviction of unretained models. Changing placement
settings does not migrate an in-flight runtime. Independent clusters remain placement units:
a model's tensor/pipeline ranks run inside one chosen cluster.

All pool members must run this protocol version and use the configured shared administration
token for signed node control and inference. An unavailable or older bridge is reported as
unavailable and cannot silently receive a load. Model files and a compatible engine must
already be installed on a candidate; the scheduler never assumes a host-specific Docker
recipe or binary can be moved to a different architecture.

Clients can use `/api/v1/pool/serve/MODEL_KEY/v1/...` on the controller, where `MODEL_KEY` is
advertised in the pool inventory. Imported legacy listeners route through this same pool.
GPU leases cover the complete streaming response. Node mutation and inference endpoints are
signed; backend API authentication is preserved separately. Pool-managed Kubernetes clients
must use this gateway for request leases and idle accounting. Manual direct-runtime traffic
remains outside the pool's request accounting. Pending loads are marked interrupted after a
controller restart rather than replayed; saved settings and retention persist.
