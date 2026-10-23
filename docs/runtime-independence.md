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
