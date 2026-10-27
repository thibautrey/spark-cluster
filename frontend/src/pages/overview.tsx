import { useState } from "react";
import { useApi, post, api, useAction } from "@/lib/api";
import { useDashboardContext } from "@/lib/context";
import type { Activity, RequestRecord } from "@/lib/types";
import type {
  ClusterSnapshot,
  HostResources,
  NodeSnapshot,
} from "@/lib/schema";
import { fresh, number, percent } from "@/lib/utils";
import {
  Panel,
  Field,
  Select,
  Empty,
  Message,
  Badge,
  Button,
  Facts,
  Meter,
  Sheet,
  Chart,
} from "@/components/common";
import { PerformanceButton } from "@/components/performance";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuCheckboxItem,
} from "@/components/ui/dropdown-menu";
import { useModelDetails } from "./models";
export function physicalHosts(nodes: NodeSnapshot[]) {
  const hosts = new Map<string, HostResources>();
  for (const node of nodes) {
    if (node.host) {
      const old = hosts.get(node.host.id);
      if (!old || Date.parse(node.host.last_seen) > Date.parse(old.last_seen))
        hosts.set(node.host.id, node.host);
    }
  }
  return [...hosts.values()];
}
export function HostSummary({ host }: { host: HostResources }) {
  return (
    <div className="host-summary">
      <b>{host.hostname}</b>
      <span>
        RAM{" "}
        {fresh(host.last_seen)
          ? `${number(host.ram_used_gib)} / ${number(host.ram_total_gib)} GiB`
          : "Unavailable"}
      </span>
      <span>
        CPU {fresh(host.last_seen) ? number(host.cpu_utilization, "%") : "—"}
      </span>
      <small>
        {host.cpu_physical_cores ? `${host.cpu_physical_cores} cores · ` : ""}
        {host.cpu_logical_cores ?? "—"} threads
      </small>
    </div>
  );
}
export function ModelActions({
  model,
  activity,
}: {
  model: string;
  activity: Activity;
}) {
  const { context } = useDashboardContext();
  const action = useAction();
  const count = activity.pools.reduce(
    (sum, p) =>
      sum +
      (
        p.resident_models ||
        (p.active_model ? [{ model_id: p.active_model }] : [])
      ).filter((r) => r.model_id === model).length,
    0,
  );
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label={`Actions for ${model}`}>
          ⋮
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent onClick={(e) => e.stopPropagation()}>
        <DropdownMenuCheckboxItem
          checked={activity.retained_models.includes(model)}
          disabled={action.busy}
          onCheckedChange={(value) =>
            void action.run(() =>
              api(
                `/api/v1/pool/models/${encodeURIComponent(model)}/retention`,
                {
                  method: "PATCH",
                  interactive: true,
                  body: { keep_loaded: value },
                },
              ),
            )
          }
        >
          Keep in memory
        </DropdownMenuCheckboxItem>
        {count > 0 && (
          <>
            <DropdownMenuItem
              disabled={count <= 1 || action.busy}
              onSelect={(e) => {
                e.preventDefault();
                void action.run(() =>
                  post("/api/v1/pool/scale", { model_id: model, delta: -1 }),
                );
              }}
            >
              − Remove instance ({count})
            </DropdownMenuItem>
            <DropdownMenuItem
              disabled={action.busy}
              onSelect={(e) => {
                e.preventDefault();
                void action.run(() =>
                  post("/api/v1/pool/scale", { model_id: model, delta: 1 }),
                );
              }}
            >
              + Add instance
            </DropdownMenuItem>
          </>
        )}
        <DropdownMenuItem
          disabled={action.busy}
          onSelect={(e) => {
            e.preventDefault();
            void action.run(
              () =>
                post(`/api/v1/pool/${count ? "unload" : "load"}`, {
                  model_id: model,
                  cluster_id: context === "all" ? null : context,
                }),
              count ? "Model unloaded." : "Load requested.",
            );
          }}
        >
          {count ? "Unload" : "Load"} model
        </DropdownMenuItem>
        <Message error={action.error}>{action.message}</Message>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
export function Overview() {
  const { snapshot, pool, snapshotError } = useDashboardContext();
  const activity = useApi<Activity>(
    "/api/v1/inference/activity?limit=12",
    2000,
  );
  const inspect = useModelDetails();
  const [history, setHistory] = useState<{ id: string; metric: string } | null>(
    null,
  );
  const nodes = snapshot?.nodes || [];
  const hosts = physicalHosts(nodes);
  const online =
    snapshot && fresh(snapshot.timestamp)
      ? nodes.filter((n) => n.status !== "offline")
      : [];
  const memoryUsed = online.reduce((s, n) => s + n.memory.used_gib, 0);
  const memoryTotal = online.reduce((s, n) => s + n.memory.total_gib, 0);
  const ramHosts = hosts.filter(
    (h) =>
      fresh(h.last_seen) && h.ram_total_gib != null && h.ram_used_gib != null,
  );
  const ramUsed = ramHosts.reduce((s, h) => s + h.ram_used_gib!, 0);
  const ramTotal = ramHosts.reduce((s, h) => s + h.ram_total_gib!, 0);
  const cpu = hosts.filter(
    (h) =>
      fresh(h.last_seen) && h.cpu_utilization != null && h.cpu_logical_cores,
  );
  const cpuCores = cpu.reduce((s, h) => s + h.cpu_logical_cores!, 0);
  const cpuLoad = cpuCores
    ? cpu.reduce((s, h) => s + h.cpu_utilization! * h.cpu_logical_cores!, 0) /
      cpuCores
    : null;
  const a = activity.data;
  const m = a?.runtime_metrics;
  const measured = m?.available && fresh(m.sampled_at);
  const rows = a ? [...a.active, ...a.recent].slice(0, 12) : [];
  const hottest = online
    .filter((n) => n.temperature_c > 0)
    .sort((a, b) => b.temperature_c - a.temperature_c)[0];
  return (
    <>
      <Message error={snapshotError} />
      <div className="metrics">
        <Panel title={measured ? "Runtime output" : "Tracked requests"}>
          <strong className="metric-value">
            {measured
              ? number(m?.decode_tokens_per_second, " tok/s")
              : number(a?.totals.requests)}
          </strong>
          <Chart
            label="Output tokens per second · minute average"
            values={
              a
                ? Array.from({ length: 40 }, (_, i) => {
                    const t =
                      Math.floor(Date.now() / 60000) * 60 - (39 - i) * 60;
                    return (
                      (a.history.find((p) => p.timestamp === t)
                        ?.output_tokens || 0) / 60
                    );
                  })
                : []
            }
          />
          <PerformanceButton>Speed history ↗</PerformanceButton>
        </Panel>
        <Panel title="GPU memory">
          <strong className="metric-value">
            {online.length ? number(memoryUsed) : "—"}
            <small> / {number(memoryTotal)} GiB</small>
          </strong>
          <Meter
            value={percent(memoryUsed, memoryTotal)}
            label="GPU memory allocated"
          />
          <p>
            Machine RAM ·{" "}
            {ramHosts.length
              ? `${number(ramUsed)} / ${number(ramTotal)} GiB`
              : "Unavailable"}
          </p>
          <Meter value={percent(ramUsed, ramTotal)} label="Machine RAM used" />
        </Panel>
        <Panel title="Machine CPU">
          <strong className="metric-value">{number(cpuLoad, "%")}</strong>
          <p className="note">
            {hosts.reduce((s, h) => s + (h.cpu_physical_cores || 0), 0)} cores ·{" "}
            {hosts.reduce((s, h) => s + (h.cpu_logical_cores || 0), 0)} threads
          </p>
          <div className="cpu-machines">
            {hosts.map((h) => (
              <div key={h.id}>
                <small>{hosts.length > 1 ? h.hostname : ""}</small>
                <div className="cpu-cores">
                  {h.cpu_cores.map((c) => (
                    <span
                      key={c.id}
                      title={`${h.hostname} · Logical CPU ${c.id}`}
                    >
                      <small>{c.id}</small>
                      <b>
                        {fresh(h.last_seen) ? number(c.utilization, "%") : "—"}
                      </b>
                      <i
                        style={{
                          width: `${fresh(h.last_seen) ? (c.utilization ?? 0) : 0}%`,
                        }}
                      />
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <Facts
        items={[
          ["Reporting nodes", `${online.length} / ${nodes.length}`],
          [
            "Average GPU load",
            online.length
              ? number(
                  online.reduce((s, n) => s + n.gpu_utilization, 0) /
                    online.length,
                  "%",
                )
              : "—",
          ],
          [
            "Memory headroom",
            online.length ? number(memoryTotal - memoryUsed, " GiB") : "—",
          ],
          [
            "Hottest GPU",
            hottest
              ? `${number(hottest.temperature_c, "°C")} · ${hottest.hostname}`
              : "Unavailable",
          ],
        ]}
      />
      <Panel
        title="Inference activity"
        actions={
          <>
            <PerformanceButton>Performance history ↗</PerformanceButton>
            <a href="#launch">Manage runtime ↗</a>
          </>
        }
      >
        <Message error={activity.error} />
        {a ? (
          <>
            <p>
              {a.active_requests
                ? `${a.active_requests} requests in progress · ${a.queued_requests} waiting for a pool`
                : `Idle · ${number(a.totals.requests)} requests recorded since ${new Date(a.since * 1000).toLocaleString()}`}
            </p>
            <Facts
              items={[
                [
                  "Requests / errors",
                  `${number(a.totals.requests)} / ${number(a.totals.failed)}`,
                ],
                [
                  "Active / waiting",
                  `${a.active_requests - a.queued_requests} / ${a.queued_requests}`,
                ],
                [
                  "Input / output tokens",
                  `${number(a.totals.input_tokens)} / ${number(a.totals.output_tokens)}`,
                ],
                [
                  "Latency p50 / p95",
                  `${number(a.latency.p50_ms, " ms")} / ${number(a.latency.p95_ms, " ms")}`,
                ],
                [
                  "First token p50 / p95",
                  `${number(a.latency.ttft_p50_ms, " ms")} / ${number(a.latency.ttft_p95_ms, " ms")}`,
                ],
                [
                  "Pools loaded / available",
                  `${a.pools.filter((p) => p.resident_models?.length || p.active_model).length} / ${a.pools.filter((p) => p.available && p.age_seconds < 15).length}`,
                ],
              ]}
            />
            <p className="note">
              {a.totals.successful} successful · {a.totals.cancelled} cancelled
              · {number(a.totals.cached_tokens)} cached input tokens ·{" "}
              {a.totals.unknown_usage} requests without token usage. Latency
              covers the latest {a.latency.sample_size} recorded requests.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {[
                      "Recent requests",
                      "Model / pool",
                      "Result",
                      "Latency",
                      "Tokens in / out",
                      "Pool memory",
                      "Actions",
                    ].map((h) => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r: RequestRecord) => {
                    const mem = r.memory_end || r.memory_start;
                    return (
                      <tr key={r.id}>
                        <td>
                          {new Date(r.started_at * 1000).toLocaleTimeString()}
                        </td>
                        <td>
                          <Button
                            variant="link"
                            onClick={() => inspect(r.model_id)}
                          >
                            {r.model_id}
                          </Button>
                          <small>
                            {a.pools.find((p) => p.id === r.cluster_id)?.name ||
                              r.cluster_id ||
                              "Unassigned"}
                          </small>
                          <PerformanceButton scope="model" entity={r.model_id}>
                            Speed & cache ↗
                          </PerformanceButton>
                        </td>
                        <td>{r.outcome || r.state}</td>
                        <td>{number(r.duration_ms ?? r.elapsed_ms, " ms")}</td>
                        <td>
                          {number(r.input_tokens)} / {number(r.output_tokens)}
                        </td>
                        <td>
                          {mem
                            ? `${number(mem.used_gib)} / ${number(mem.total_gib)} GiB`
                            : "Unavailable"}
                        </td>
                        <td>
                          <ModelActions model={r.model_id} activity={a} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {!rows.length && (
              <Empty>
                Tracking is ready. Requests appear here as clients use the pool
                endpoints.
              </Empty>
            )}
            <details>
              <summary>Pool usage and tracking details</summary>
              {a.pools.map((p) => (
                <p key={p.id}>
                  <PerformanceButton scope="pool" entity={p.id}>
                    {p.name}
                  </PerformanceButton>{" "}
                  ·{" "}
                  {p.resident_models?.map((r) => r.model_id).join(", ") ||
                    p.active_model ||
                    "No model loaded"}{" "}
                  ·{" "}
                  {p.age_seconds < 15 && p.available
                    ? `${number(p.machines.reduce((s, m) => s + (m.memory_used_gib || 0), 0))} / ${number(p.machines.reduce((s, m) => s + (m.memory_total_gib || 0), 0))} GiB`
                    : "Resource sample unavailable"}{" "}
                  · {p.busy ? "busy" : "idle"}
                </p>
              ))}
              {a.groups.map((g, i) => (
                <p key={i}>
                  {g.model_id} · {number(g.requests)} requests ·{" "}
                  {number(g.input_tokens)} input / {number(g.output_tokens)}{" "}
                  output tokens
                </p>
              ))}
              <p className="note">
                {a.coverage} Memory is a shared pool sample. Records retain no
                prompts, responses or credentials.{" "}
                {a.persistence.healthy
                  ? "Journal healthy"
                  : "Journal write failing"}{" "}
                · {a.persistence.pending} pending · {a.persistence.dropped}{" "}
                dropped.
              </p>
            </details>
          </>
        ) : (
          <Empty>Loading recorded activity…</Empty>
        )}
      </Panel>
      <Panel
        title="Node resources"
        actions={<a href="#topology">View topology ↗</a>}
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {[
                  "Node",
                  "Status",
                  "GPU load",
                  "GPU memory",
                  "Temperature",
                ].map((x) => (
                  <th key={x}>{x}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {nodes.map((n) => {
                const available = online.includes(n);
                return (
                  <tr key={n.id}>
                    <th>
                      {n.hostname}
                      <small>
                        {available && n.loaded_model_ids.length
                          ? `Model · ${n.loaded_model_ids.join(", ")}`
                          : ""}
                      </small>
                    </th>
                    <td>
                      <Badge good={available && n.status === "healthy"}>
                        {available ? n.status : "stale"}
                      </Badge>
                    </td>
                    {[
                      ["gpu", number(n.gpu_utilization, "%")],
                      [
                        "memory",
                        `${number(n.memory.used_gib)} / ${number(n.memory.total_gib)} GiB`,
                      ],
                      ["temperature", number(n.temperature_c, "°C")],
                    ].map(([metric, label]) => (
                      <td key={metric}>
                        {available ? (
                          <Button
                            variant="ghost"
                            onClick={() => setHistory({ id: n.id, metric })}
                            aria-label={`View ${metric} history for ${n.hostname}`}
                          >
                            {label} ↗
                          </Button>
                        ) : (
                          "—"
                        )}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!nodes.length && <Empty>No nodes in this pool yet.</Empty>}
        <p className="note">
          Snapshot{" "}
          {snapshot
            ? new Date(snapshot.timestamp).toLocaleString()
            : "unavailable"}
          {snapshot && !fresh(snapshot.timestamp)
            ? " · stale; current resources unknown"
            : ""}
        </p>
      </Panel>
      {history && (
        <ResourceHistory selection={history} close={() => setHistory(null)} />
      )}
    </>
  );
}
function ResourceHistory({
  selection,
  close,
}: {
  selection: { id: string; metric: string };
  close: () => void;
}) {
  const q = useApi<ClusterSnapshot[]>("/api/v1/history?minutes=30");
  const values =
    q.data?.flatMap((s) => {
      const n = s.nodes.find((n) => n.id === selection.id);
      return n
        ? [
            {
              time: s.timestamp,
              value:
                selection.metric === "gpu"
                  ? n.gpu_utilization
                  : selection.metric === "memory"
                    ? n.memory.used_gib
                    : n.temperature_c,
            },
          ]
        : [];
    }) || [];
  const data = values.map((p) => p.value);
  return (
    <Sheet
      open
      onOpenChange={(v) => {
        if (!v) close();
      }}
      title={`${selection.metric} history`}
      description="Last 30 minutes"
    >
      <Message error={q.error} />
      {q.isLoading ? (
        <p>Loading resource history…</p>
      ) : (
        <>
          <Facts
            items={[
              ["Current", number(data.at(-1))],
              ["Minimum", data.length ? number(Math.min(...data)) : "—"],
              [
                "Average",
                data.length
                  ? number(data.reduce((a, b) => a + b, 0) / data.length)
                  : "—",
              ],
              ["Maximum", data.length ? number(Math.max(...data)) : "—"],
            ]}
          />
          <Chart
            label={`${selection.metric} · ${data.length} samples`}
            values={data}
          />
          <p className="note">
            {values.length
              ? `${new Date(values[0].time).toLocaleString()} → ${new Date(values.at(-1)!.time).toLocaleString()}`
              : "No recorded samples"}
          </p>
        </>
      )}
    </Sheet>
  );
}
