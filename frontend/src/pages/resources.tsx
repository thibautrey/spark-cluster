import { useState } from "react";
import { useDashboardContext } from "@/lib/context";
import { api, post, useAction, useApi } from "@/lib/api";
import type { CachePrefixRecord, InferenceTelemetry } from "@/lib/schema";
import { fresh, number, percent } from "@/lib/utils";
import {
  Panel,
  Empty,
  Meter,
  Message,
  Field,
  Input,
  Textarea,
  Button,
  Badge,
} from "@/components/common";
import { physicalHosts, HostSummary } from "./overview";
import { PerformanceButton } from "@/components/performance";
export function Topology() {
  const { snapshot, clusters } = useDashboardContext();
  const active = clusters?.clusters.find(
    (c) => c.id === clusters.active_cluster_id,
  );
  return (
    <Panel title={active?.name || "Pool topology"}>
      {physicalHosts(snapshot?.nodes || []).map((h) => (
        <HostSummary key={h.id} host={h} />
      ))}
      <div className="card-grid">
        {snapshot?.nodes.map((n, i) => (
          <article className="node-card" key={n.id}>
            <header>
              <Badge good={n.status === "healthy" && fresh(snapshot.timestamp)}>
                {fresh(snapshot.timestamp) ? n.status : "stale"}
              </Badge>
              <PerformanceButton
                scope="gpu"
                entity={JSON.stringify([
                  active?.id || "",
                  active?.gpus[i]?.uuid || n.id,
                ])}
              />
            </header>
            <h3>
              {active?.kind === "local-docker"
                ? `GPU ${active.gpus[i]?.index ?? i}`
                : n.hostname}
            </h3>
            <p className="note">
              {active?.kind === "local-docker"
                ? active.gpus[i]?.name
                : `${n.role.toUpperCase()} · ${n.address}`}
            </p>
            <p>
              GPU {number(n.gpu_utilization, "%")} · {number(n.memory.used_gib)}{" "}
              / {number(n.memory.total_gib)} GiB
            </p>
            <Meter
              value={percent(n.memory.used_gib, n.memory.total_gib)}
              label="GPU memory"
            />
            <p>
              {number(n.output_tokens_per_second, " tok/s")} ·{" "}
              {number(n.temperature_c, "°C")}
            </p>
          </article>
        ))}
      </div>
      {!snapshot?.nodes.length && <Empty>Waiting for telemetry</Empty>}
      <p className="note">
        {active?.kind === "local-docker" ? "1 host · " : ""}
        {snapshot?.nodes.length || 0} accelerators
      </p>
    </Panel>
  );
}
export function memorySegments(memory: {
  total_gib: number;
  used_gib: number;
  model_gib: number;
  kv_cache_gib: number;
  kv_cache_source: string;
}) {
  const total = Math.max(0, memory.total_gib);
  const model = Math.min(total, Math.max(0, memory.model_gib));
  const kv =
    memory.kv_cache_source !== "unavailable"
      ? Math.min(total - model, Math.max(0, memory.kv_cache_gib))
      : 0;
  const other = Math.min(
    total - model - kv,
    Math.max(0, memory.used_gib - model - kv),
  );
  return [
    ["Model", model],
    ["KV cache", kv],
    ["Used", other],
    ["Free", Math.max(0, total - model - kv - other)],
  ] as [string, number][];
}
export function Memory() {
  const { snapshot } = useDashboardContext();
  return (
    <Panel title="Memory allocation">
      <div className="card-grid">
        {physicalHosts(snapshot?.nodes || []).map((h) => (
          <article className="node-card" key={h.id}>
            <HostSummary host={h} />
            <Meter
              label="Machine RAM used"
              value={
                fresh(h.last_seen)
                  ? percent(h.ram_used_gib || 0, h.ram_total_gib || 0)
                  : 0
              }
            />
            <p className="note">
              {fresh(h.last_seen)
                ? `${number(h.ram_available_gib)} GiB available`
                : "Waiting for host telemetry"}
            </p>
          </article>
        ))}
      </div>
      <div className="stack">
        {snapshot?.nodes.map((n, i) => (
          <div key={n.id}>
            <div className="split">
              <b>
                GPU {i} · {n.hostname}
              </b>
              <span>
                {fresh(snapshot.timestamp)
                  ? `${number(n.memory.used_gib)} / ${number(n.memory.total_gib)} GiB`
                  : "Unavailable"}
              </span>
            </div>
            <div
              className="memory-bar"
              role="img"
              aria-label={`${n.hostname}: ${number(n.memory.used_gib)} of ${number(n.memory.total_gib)} GiB used`}
            >
              {memorySegments(n.memory).map(([label, value], i) => (
                <i
                  key={label}
                  style={{
                    flex: value,
                    background: ["#b0e68b", "#71b9e8", "#b693e4", "#1b2633"][i],
                  }}
                  title={`${label} ${number(value)} GiB`}
                />
              ))}
            </div>
            <div className="actions note">
              {memorySegments(n.memory)
                .filter(([, v]) => v > 0)
                .map(([label, value]) => (
                  <span key={label}>
                    {label} · {number(value)} GiB
                  </span>
                ))}
            </div>
          </div>
        ))}
      </div>
      {!snapshot?.nodes.length && <Empty>Waiting for telemetry</Empty>}
    </Panel>
  );
}
export function Cache() {
  const { snapshot, launchSettings } = useDashboardContext();
  const records = useApi<CachePrefixRecord[]>(
    "/api/v1/cache/prefixes?limit=20",
    5000,
  );
  const telemetry = useApi<InferenceTelemetry>("/api/v1/telemetry", 5000);
  const action = useAction();
  const [model, setModel] = useState("");
  const [tokenizer, setTokenizer] = useState("");
  const [prompt, setPrompt] = useState("");
  const [priority, setPriority] = useState(50);
  const tiers =
    snapshot?.cache_tiers.filter(
      (t) => !t.name.toLowerCase().includes("model storage"),
    ) || [];
  return (
    <>
      <Panel
        title="KV cache"
        actions={<PerformanceButton>Cache history ↗</PerformanceButton>}
      >
        {tiers.length ? (
          tiers.map((t) => (
            <div className="stack" key={t.name}>
              <div className="split">
                <b>{t.name}</b>
                <span>
                  {number(t.used_gib)} / {number(t.capacity_gib)} GiB
                </span>
              </div>
              <Meter
                value={percent(t.used_gib, t.capacity_gib)}
                label={`${t.name} usage`}
              />
              <p className="note">
                {number(percent(t.hits, t.hits + t.misses), "%")} hit ·{" "}
                {number(t.read_latency_ms, " ms")}
              </p>
            </div>
          ))
        ) : (
          <Empty>
            KV cache metrics unavailable. Metrics appear after a runtime starts.{" "}
            <a href="#launch">Launch a model ↗</a>
          </Empty>
        )}
      </Panel>
      {(telemetry.data?.available || !!records.data?.length) && (
        <Panel>
          <details>
            <summary>Prewarm reusable context · Advanced</summary>
            <p className="note">
              Raw prompts are sent only to your runtime and are never stored by
              the controller.
            </p>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void action.run(async () => {
                  await post("/api/v1/cache/prefixes/prewarm", {
                    model_id: model,
                    tokenizer_id: tokenizer || model,
                    prompt,
                    engine_id:
                      launchSettings.engine_id === "auto"
                        ? "vllm"
                        : launchSettings.engine_id,
                    priority,
                    engine_config: {
                      context_length: launchSettings.context_length,
                      kv_cache_dtype: launchSettings.kv_cache_dtype,
                    },
                  });
                  setPrompt("");
                }, "Prefix prewarmed.");
              }}
            >
              <div className="form-grid">
                <Field label="Model id">
                  <Input
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    required
                  />
                </Field>
                <Field label="Tokenizer id">
                  <Input
                    value={tokenizer}
                    onChange={(e) => setTokenizer(e.target.value)}
                  />
                </Field>
                <Field label="Priority">
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={priority}
                    onChange={(e) => setPriority(+e.target.value)}
                  />
                </Field>
              </div>
              <Field label="Reusable prefix">
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  required
                />
              </Field>
              <Button disabled={action.busy}>Prewarm prefix</Button>
              <Message error={action.error}>{action.message}</Message>
            </form>
            {records.data?.map((r) => (
              <article className="list-row" key={r.id}>
                <div>
                  <b>{r.model_id}</b>
                  <small>
                    {number(r.prompt_tokens)} tokens · priority {r.priority} ·{" "}
                    {r.prewarm_count} prewarms · TTFT{" "}
                    {number(r.last_ttft_ms, " ms")}
                  </small>
                </div>
                <Badge>{r.state}</Badge>
              </article>
            ))}
          </details>
        </Panel>
      )}
    </>
  );
}
