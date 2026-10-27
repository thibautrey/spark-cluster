# React dashboard

The dashboard is a React 19 application written in strict TypeScript. Its shadcn/ui components use Radix primitives, Tailwind CSS and class-variance-authority. FastAPI continues to serve the production bundle at `/` and `/assets`; REST endpoints, WebSocket protocols, inference routing and backend storage retain their existing contracts.

## Development and packaging

Run these commands from the main checkout:

```sh
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix frontend test
npm --prefix frontend run test:ui
.venv/bin/pytest -q
```

The production build is packaged in `src/spark_cluster/static` for Python installations. Docker rebuilds it in a separate Node stage; Node is not required in the Python runtime image. `npm --prefix frontend run dev` provides a Vite development server proxying `/api` and its WebSockets to `127.0.0.1:8080`.

Regenerate the Pydantic response contracts after a backend schema change:

```sh
.venv/bin/python scripts/export-ui-schema.py
```

Keep API credentials out of source, build arguments and bundles. The admin token uses the existing origin-scoped browser key and is stored only after server validation. Background reads never open authentication dialogs; actions can request authentication and retry after validation.

## Parity audit

| Route | Preserved behavior |
| --- | --- |
| Overview | Snapshot stream, request activity, model residency and instance actions, GPU and host CPU/RAM, stale telemetry, resource and performance history |
| Compute pools | Local GPU discovery, pool selection, existing bridge connection, discovery, SSH bootstrap, topology and fabric inspection, readiness, explicit installation approval, saved jobs and recovery, native Mac registration |
| Model library | Hub search/sort/filter/pagination, capacity batches, recommendation evidence, details/README/files, download fit, active transfer controls, retries, local inventory/deletion, HF token |
| Launch | Model-centric capacity, GPU budgets, automatic placement, filtering, load/unload, residency and pool preferences, saved advanced settings, performance planning, local fit, guided prepare/launch, retry, benchmarks |
| Runtime readiness | Engine readiness, prepare/update workflows, per-node state, native hosts, published versions, cache/download/install/remove packages |
| Logs | History and WebSocket updates, instance and level filters, pause/resume/clear, bounded buffer |
| Topology | Active pool identity, host resources, per-accelerator telemetry and history links |
| Memory | Deduplicated host RAM, model/KV/runtime/free memory, unavailable KV handling |
| KV cache | Tier occupancy and hit rate, prewarm and recorded prefixes |
| Inference API | Stable pool OpenAI endpoint, model/chat/embedding routes and MCP endpoint |

React owns all application rendering and event handling. Sanitized Markdown is the only HTML fragment rendered from a string. Dialogs and action menus use Radix focus management and keyboard handling. Data reads use React Query cancellation/cache isolation; WebSockets close and reconnect with lifecycle cleanup.

Live mutation smoke tests must avoid unloading, replacing or deleting production models. Exercise destructive paths against fixtures, then verify the deployed bundle, all ten routes, controller health and inference routing separately.
