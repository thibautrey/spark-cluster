import { useEffect, useRef, useState } from "react";
import { api, post, useAction, useApi } from "@/lib/api";
import { useDashboardContext } from "@/lib/context";
import type {
  RuntimeEngineCollection,
  RuntimeGuide,
  RuntimeWorkflowJob,
  ModelLaunchRequest,
  RuntimeReadiness,
  RuntimeTerminalTicket,
} from "@/lib/schema";
import type { Library, PackageRuntime } from "@/lib/types";
import { bytes, number } from "@/lib/utils";
import {
  Panel,
  Field,
  Select,
  Input,
  Button,
  Message,
  Empty,
  Sheet,
  Copy,
  Badge,
  Meter,
  Facts,
} from "@/components/common";
const base = "/api/v1/runtime-library";
const management = "/api/v1/runtime-management";
const platformLabels: Record<string, string> = {
  "linux/arm64": "Linux ARM64 · DGX Spark",
  "linux/amd64": "Linux x86-64 · NVIDIA",
  "darwin/arm64": "macOS · Apple Silicon",
  "darwin/amd64": "macOS · Intel",
  "windows/amd64": "Windows · x64",
  "windows/arm64": "Windows · ARM64",
};
export function Runtimes() {
  const { clusters, context } = useDashboardContext();
  const cluster =
    clusters?.clusters.find((c) => c.id === context) ||
    (context === "all" && clusters?.clusters.length === 1
      ? clusters.clusters[0]
      : null);
  const library = useApi<Library>(base, 5000);
  const engines = useApi<RuntimeEngineCollection>(
    cluster
      ? `${management}/engines?cluster_id=${encodeURIComponent(cluster.id)}`
      : null,
    5000,
  );
  const [tab, setTab] = useState("engines");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [guide, setGuide] = useState<RuntimeGuide | null>(null);
  const action = useAction();
  return (
    <>
      <Panel
        title={cluster?.name || "Runtime readiness"}
        actions={
          <Button
            variant="outline"
            disabled={action.busy}
            onClick={() =>
              void action.run(() =>
                api("/api/v1/controller/verify", { interactive: true }),
              )
            }
          >
            Administrator access
          </Button>
        }
      >
        <p className="note">
          {cluster
            ? `${cluster.gpu_count} GPUs · Automatic placement`
            : library.data?.hosts.length
              ? "Native host ready. Select a verified native package for the registered host."
              : "Choose one compute pool to inspect runtime readiness."}
        </p>
        <div className="toolbar">
          <Button
            variant={tab === "engines" ? "secondary" : "ghost"}
            onClick={() => setTab("engines")}
          >
            Runtimes · {engines.data?.engines.length || 0}
          </Button>
          <Button
            variant={tab === "packages" ? "secondary" : "ghost"}
            onClick={() => setTab("packages")}
          >
            Packages · {library.data?.runtimes.length || 0}
          </Button>
        </div>
      </Panel>
      <Message error={action.error || library.error} />
      {tab === "engines" && (
        <Panel
          title="Runtime engines"
          actions={
            <Button
              variant="outline"
              disabled={action.busy || !cluster}
              onClick={() =>
                void action.run(() =>
                  api(
                    `${management}/engines?cluster_id=${encodeURIComponent(cluster!.id)}`,
                    { interactive: true },
                  ),
                )
              }
            >
              Revalidate
            </Button>
          }
        >
          <Message error={engines.error} />
          <div className="toolbar">
            <Input
              aria-label="Search runtimes"
              placeholder="Search runtimes…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Select
              aria-label="Runtime state"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            >
              <option value="all">All states</option>
              <option value="ready">Ready</option>
              <option value="attention">Needs preparation</option>
            </Select>
          </div>
          {engines.data?.engines
            .filter((e) => e.capability === "managed_launch" || e.image)
            .filter(
              (e) =>
                `${e.name} ${e.id}`
                  .toLowerCase()
                  .includes(query.toLowerCase()) &&
                (filter === "all" || (filter === "ready" ? e.ready : !e.ready)),
            )
            .map((e) => (
              <article className="runtime-card" key={e.id}>
                <header>
                  <div>
                    <h3>{e.name}</h3>
                    <small>
                      {e.nodes.find((n) => n.installed_version)
                        ?.installed_version || "Configuration required"}
                    </small>
                  </div>
                  <Badge good={e.ready}>
                    {e.ready
                      ? "✓ Ready to launch"
                      : e.state.replaceAll("_", " ")}
                  </Badge>
                </header>
                <p>{e.message}</p>
                <p className="note">
                  {e.nodes_ready} / {e.nodes_total} targets ready
                </p>
                {e.active_job_id && (
                  <Meter
                    value={e.active_job_progress || 0}
                    label={`${e.name} preparation`}
                  />
                )}
                <details>
                  <summary>Node & technical details</summary>
                  {e.nodes.map((n) => (
                    <p key={n.address}>
                      {n.address} · {n.status} · {n.message}
                    </p>
                  ))}
                  <code>{e.image}</code>
                </details>
                <footer className="actions">
                  <Button
                    disabled={action.busy || !!e.active_job_id}
                    onClick={() =>
                      void action.run(async () =>
                        setGuide(
                          await api<RuntimeGuide>(
                            `${management}/guide?${new URLSearchParams({ cluster_id: cluster!.id, engine_id: e.id })}`,
                            { interactive: true },
                          ),
                        ),
                      )
                    }
                  >
                    {e.ready ? "Review / update runtime" : `Set up ${e.name}`}
                  </Button>
                  {e.prepare_supported && !e.ready && (
                    <Button
                      variant="outline"
                      disabled={action.busy || !!e.active_job_id}
                      onClick={() =>
                        void action.run(() =>
                          post(`${management}/prepare`, {
                            cluster_id: cluster!.id,
                            engine_id: e.id,
                          }),
                        )
                      }
                    >
                      Prepare on this pool
                    </Button>
                  )}
                  {e.ready && <a href="#launch">Launch a model ↗</a>}
                </footer>
              </article>
            ))}
          {!cluster && (
            <Empty>
              Choose one compute pool, or open Packages to manage native hosts.
            </Empty>
          )}
        </Panel>
      )}
      {(tab === "packages" || !clusters?.clusters.length) && library.data && (
        <PackageLibrary
          library={library.data}
          clusterId={cluster?.id}
          clusterPlatform={
            engines.data?.platform ||
            (cluster?.hardware === "dgx-spark" ? "linux/arm64" : "linux/amd64")
          }
        />
      )}
      {guide && <RuntimeWorkflow guide={guide} close={() => setGuide(null)} />}
    </>
  );
}
export function RuntimeWorkflow({
  guide,
  launch,
  close,
}: {
  guide: RuntimeGuide;
  launch?: ModelLaunchRequest;
  close: () => void;
}) {
  const [id, setId] = useState<string | null>(null);
  const job = useApi<RuntimeWorkflowJob>(
    id ? `${management}/workflows/${encodeURIComponent(id)}` : null,
    1000,
  );
  const action = useAction();
  const active = job.data && ["queued", "running"].includes(job.data.status);
  return (
    <Sheet
      open
      onOpenChange={(v) => {
        if (!v) close();
      }}
      title={launch ? "Prepare and launch" : `Set up ${guide.engine_name}`}
      description={guide.summary}
    >
      <Facts
        items={[
          ["Runtime", guide.engine_name],
          ["Version", guide.version || "Unavailable"],
          ["Compute pool", guide.cluster_name],
        ]}
      />
      {(job.data?.steps || guide.steps).map((s, i) => (
        <article className="list-row" key={s.id || i}>
          <span className="step-number">{i + 1}</span>
          <div>
            <b>{s.title}</b>
            <small>{"message" in s ? s.message : s.description}</small>
          </div>
          <Badge>{"status" in s ? s.status : "pending"}</Badge>
        </article>
      ))}
      {job.data && (
        <>
          <Meter value={job.data.progress} label="Runtime preparation" />
          <p>
            {job.data.status} · {number(job.data.progress, "%")}
          </p>
          {job.data.response_preview && (
            <p>Test response: {job.data.response_preview}</p>
          )}
        </>
      )}
      <Message error={action.error || job.error || job.data?.error} />
      {!guide.compatible && (
        <p className="error">
          {[...guide.blockers, ...guide.alternatives].join(" ")}
        </p>
      )}
      <details>
        <summary>Technical details</summary>
        <pre>
          {guide.image}
          {guide.digest ? `@${guide.digest}` : ""}
        </pre>
      </details>
      {launch && (
        <RuntimeTerminal model={launch} clusterId={guide.cluster_id} />
      )}
      <div className="actions">
        <Button variant="outline" onClick={close}>
          Later
        </Button>
        <Button
          disabled={
            action.busy ||
            !!active ||
            job.data?.status === "completed" ||
            !guide.compatible
          }
          onClick={() =>
            void action.run(async () => {
              const response =
                job.data?.status === "failed"
                  ? await post<RuntimeWorkflowJob>(
                      `${management}/workflows/${job.data.id}/retry`,
                    )
                  : await post<RuntimeWorkflowJob>(`${management}/workflows`, {
                      cluster_id: guide.cluster_id,
                      engine_id: guide.engine_id,
                      model_id: launch?.model_id || null,
                      launch: launch || null,
                    });
              setId(response.id);
            })
          }
        >
          {job.data?.status === "failed"
            ? "Retry failed step"
            : job.data?.status === "completed"
              ? "Ready"
              : active
                ? "Preparing…"
                : launch
                  ? "Prepare and launch"
                  : guide.update_available
                    ? "Update runtime"
                    : `Set up ${guide.engine_name}`}
        </Button>
      </div>
    </Sheet>
  );
}
function RuntimeTerminal({
  model,
  clusterId,
}: {
  model: ModelLaunchRequest;
  clusterId: string;
}) {
  const [open, setOpen] = useState(false);
  const report = useApi<RuntimeReadiness>(null);
  const [nodes, setNodes] = useState<RuntimeReadiness["nodes"]>([]);
  const [address, setAddress] = useState("");
  const [output, setOutput] = useState("");
  const [connected, setConnected] = useState(false);
  const [command, setCommand] = useState("");
  const action = useAction();
  const socket = useRef<WebSocket | null>(null);
  const terminal = useRef<HTMLPreElement>(null);
  useEffect(
    () => () => {
      socket.current?.close();
    },
    [],
  );
  useEffect(() => {
    if (terminal.current)
      terminal.current.scrollTop = terminal.current.scrollHeight;
  }, [output]);
  async function connect() {
    socket.current?.close();
    const ticket = await post<RuntimeTerminalTicket>(
      `${management}/terminal-ticket`,
      { address },
    );
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${management}/terminal/${encodeURIComponent(ticket.ticket)}`,
    );
    socket.current = ws;
    setOutput(`Connecting to ${address}…\n`);
    ws.onopen = () => setConnected(true);
    ws.onmessage = (e) => {
      const payload = JSON.parse(e.data) as { type: string; data: string };
      if (payload.type === "output")
        setOutput((v) =>
          (
            v +
            payload.data
              .replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "")
              .replace(/\r/g, "")
          ).slice(-100000),
        );
    };
    ws.onclose = () => {
      setConnected(false);
      setOutput((v) => v + "\n[terminal disconnected]\n");
    };
  }
  return (
    <details
      open={open}
      onToggle={(e) => {
        setOpen(e.currentTarget.open);
        if (e.currentTarget.open && !nodes.length)
          void action.run(async () => {
            const r = await post<RuntimeReadiness>(`${management}/readiness`, {
              model_id: model.model_id,
              engine_id: model.engine_id,
              cluster_id: clusterId,
            });
            setNodes(r.nodes);
            setAddress(r.nodes[0]?.address || "");
          });
      }}
    >
      <summary>Open terminal & node details</summary>
      {nodes.map((n) => (
        <p key={n.address}>
          {n.address} · {n.status} · {n.message}
        </p>
      ))}
      <div className="toolbar">
        <Select
          value={address}
          aria-label="Terminal node"
          onChange={(e) => setAddress(e.target.value)}
        >
          {nodes.map((n) => (
            <option key={n.address} value={n.address}>
              {n.address} · {n.role}
            </option>
          ))}
        </Select>
        <Button
          type="button"
          variant="outline"
          disabled={!address || action.busy}
          onClick={() => void action.run(connect)}
        >
          Connect terminal
        </Button>
        <Button
          type="button"
          variant="outline"
          disabled={!connected}
          onClick={() =>
            socket.current?.send(
              JSON.stringify({ type: "input", data: "\u0003" }),
            )
          }
        >
          Interrupt
        </Button>
      </div>
      <Message error={action.error} />
      <pre className="terminal" ref={terminal}>
        {output}
      </pre>
      <Input
        aria-label="Terminal command"
        value={command}
        disabled={!connected}
        onChange={(e) => setCommand(e.target.value)}
        onKeyDown={(e) => {
          if (
            e.key === "Enter" &&
            socket.current?.readyState === WebSocket.OPEN
          ) {
            e.preventDefault();
            socket.current.send(
              JSON.stringify({ type: "input", data: command + "\n" }),
            );
            setCommand("");
          }
        }}
      />
    </details>
  );
}
function PackageLibrary({
  library,
  clusterId,
  clusterPlatform,
}: {
  library: Library;
  clusterId?: string;
  clusterPlatform: string;
}) {
  const [target, setTarget] = useState(
    clusterId
      ? `pool:${clusterId}`
      : library.hosts.length === 1
        ? `host:${library.hosts[0].id}`
        : "",
  );
  const [selected, setSelected] = useState<PackageRuntime | null>(null);
  const [version, setVersion] = useState("");
  const [cursor, setCursor] = useState("");
  const [remove, setRemove] = useState<{ kind: string; id: string } | null>(
    null,
  );
  const action = useAction();
  const platform = target.startsWith("host:")
    ? library.hosts.find((h) => h.id === target.slice(5))?.platform || ""
    : target
      ? clusterPlatform
      : "";
  const versions = useApi<{
    versions: { version: string; platforms: string[] }[];
    next_cursor: string | null;
    ordering: string;
    stale: boolean;
  }>(
    selected && platform
      ? `${base}/runtimes/${encodeURIComponent(selected.id)}/versions?${new URLSearchParams({ platform, cursor })}`
      : null,
  );
  useEffect(() => {
    setVersion("");
    setCursor("");
  }, [selected?.id, target]);
  useEffect(() => {
    if (!selected && library.runtimes.length) {
      setSelected(
        library.runtimes.find((r) => r.platforms.includes(platform)) ||
          library.runtimes[0],
      );
    }
  }, [library.runtimes, platform, selected]);
  return (
    <Panel title="Runtime packages">
      <p className="note">{library.message}</p>
      <Message error={action.error}>{action.message}</Message>
      <Field label="Package target">
        <Select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="">Select a compute pool or native host</option>
          {clusterId && (
            <option value={`pool:${clusterId}`}>Current compute pool</option>
          )}
          {library.hosts.map((h) => (
            <option key={h.id} value={`host:${h.id}`}>
              Standalone host · {h.name}
            </option>
          ))}
        </Select>
      </Field>
      <p className="note">
        {platformLabels[platform] || "Select a package target"}
      </p>
      <div className="package-layout">
        <div className="stack">
          {library.runtimes.map((r) => (
            <Button
              variant={r.id === selected?.id ? "secondary" : "outline"}
              key={r.id}
              onClick={() => setSelected(r)}
            >
              {r.name}
            </Button>
          ))}
        </div>
        {selected && (
          <div className="stack">
            <h3>{selected.name}</h3>
            <p>{selected.description}</p>
            <p className="note">
              {selected.hardware} ·{" "}
              {selected.launch_adapter
                ? "Managed launch"
                : selected.native_repository
                  ? "Package only"
                  : "No launch adapter"}
            </p>
            <a
              href={selected.project_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Project ↗
            </a>
            <p className="note">{selected.note}</p>
            <Message error={versions.error} />
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void action.run(
                  () =>
                    post(`${base}/downloads`, {
                      runtime_id: selected.id,
                      version: version.trim(),
                      platform,
                    }),
                  "Package download queued.",
                );
              }}
            >
              <Field label="Published versions">
                <Select
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                >
                  <option value="">Select a version</option>
                  {versions.data?.versions.map((v) => (
                    <option key={v.version} value={v.version}>
                      {v.version} · {v.platforms.join(", ")}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Version or immutable reference">
                <Input
                  value={version}
                  onChange={(e) => setVersion(e.target.value)}
                  required
                />
              </Field>
              <div className="actions">
                <Button
                  type="submit"
                  disabled={
                    action.busy ||
                    !platform ||
                    !selected.platforms.includes(platform) ||
                    (platform.startsWith("linux/") &&
                      !library.download_available)
                  }
                >
                  Download to controller cache
                </Button>
                {versions.data?.next_cursor && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setCursor(versions.data!.next_cursor!)}
                  >
                    More versions
                  </Button>
                )}
              </div>
              <p className="note">
                {versions.data?.ordering}
                {versions.data?.stale ? " · cached, source unavailable" : ""}
              </p>
            </form>
          </div>
        )}
      </div>
      <details>
        <summary>
          Cached packages · {library.artifacts.length} ·{" "}
          {bytes(library.artifacts.reduce((s, a) => s + a.archive_bytes, 0))}
        </summary>
        {library.artifacts.map((a) => (
          <article className="list-row" key={a.id}>
            <div>
              <b>
                {a.runtime_id} · {a.version}
              </b>
              <small>
                {platformLabels[a.platform] || a.platform} · {a.artifact_kind} ·{" "}
                {bytes(a.archive_bytes)} · cached only
              </small>
            </div>
            <div className="actions">
              <Button
                variant="outline"
                disabled={action.busy || !target || a.platform !== platform}
                onClick={() =>
                  void action.run(() =>
                    post(`${base}/installs`, {
                      artifact_id: a.id,
                      ...(target.startsWith("host:")
                        ? { host_id: target.slice(5) }
                        : { cluster_id: target.slice(5) }),
                    }),
                  )
                }
              >
                Install package
              </Button>
              <Button
                variant="outline"
                disabled={action.busy}
                onClick={() => setRemove({ kind: "artifacts", id: a.id })}
              >
                Remove cache
              </Button>
            </div>
          </article>
        ))}
      </details>
      <details>
        <summary>Package activity</summary>
        {library.jobs.slice(0, 12).map((j) => (
          <article className="list-row" key={j.id}>
            <div>
              <b>
                {j.operation} · {j.request.runtime_id || j.runtime_id}
              </b>
              <small>
                {j.request.version} · {j.message}
              </small>
            </div>
            <Badge>
              {j.status} · {number(j.progress, "%")}
            </Badge>
          </article>
        ))}
      </details>
      <details>
        <summary>Native hosts · {library.hosts.length}</summary>
        {library.hosts.map((h) => (
          <article className="list-row" key={h.id}>
            <div>
              <b>{h.name}</b>
              <small>
                {platformLabels[h.platform] || h.platform} · {h.username}@
                {h.address}
              </small>
            </div>
            <Button
              variant="outline"
              onClick={() => setRemove({ kind: "hosts", id: h.id })}
            >
              Remove
            </Button>
          </article>
        ))}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const form = e.currentTarget;
            const data = Object.fromEntries(new FormData(form));
            void action.run(async () => {
              await post(`${base}/hosts`, data);
              form.reset();
            }, "Host registered.");
          }}
        >
          <div className="form-grid">
            {[
              ["name", "Host name"],
              ["address", "Address"],
              ["username", "SSH username"],
              ["fingerprint", "Pinned SSH host fingerprint"],
            ].map(([name, label]) => (
              <Field label={label} key={name}>
                <Input name={name} required />
              </Field>
            ))}
          </div>
          <Button disabled={action.busy}>Register trusted host</Button>
        </form>
      </details>
      <Sheet
        open={!!remove}
        onOpenChange={(v) => {
          if (!v) setRemove(null);
        }}
        title={
          remove?.kind === "hosts"
            ? "Remove this native host?"
            : "Remove this package cache?"
        }
        description="Installed runtime images and packages on the machines remain available."
      >
        <Message error={action.error} />
        <div className="actions">
          <Button variant="outline" onClick={() => setRemove(null)}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={action.busy}
            onClick={() =>
              void action.run(async () => {
                await api(
                  `${base}/${remove!.kind}/${encodeURIComponent(remove!.id)}`,
                  { method: "DELETE", interactive: true },
                );
                setRemove(null);
              })
            }
          >
            Remove
          </Button>
        </div>
      </Sheet>
    </Panel>
  );
}
