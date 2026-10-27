import { Component, useEffect, type ReactNode, type ErrorInfo } from "react";
import {
  Activity,
  Boxes,
  Database,
  Layers,
  Library,
  MemoryStick,
  Network,
  Radio,
  ScrollText,
  Server,
  Zap,
} from "lucide-react";
import { routes, useDashboardContext, type Route } from "./lib/context";
import {
  Panel,
  Field,
  Select,
  Copy,
  Message,
  Badge,
  Button,
} from "./components/common";
import { Overview } from "./pages/overview";
import { ModelDetailsProvider, Models } from "./pages/models";
import { Launch } from "./pages/launch";
import { Runtimes } from "./pages/runtimes";
import { Setup } from "./pages/setup";
import { Logs } from "./pages/logs";
import { Topology, Memory, Cache } from "./pages/resources";
const icons = {
  setup: Server,
  overview: Activity,
  api: Radio,
  models: Library,
  runtimes: Boxes,
  launch: Zap,
  logs: ScrollText,
  topology: Network,
  cache: Database,
  memory: MemoryStick,
};
class ErrorBoundary extends Component<
  { children: ReactNode; route: Route },
  { error: boolean }
> {
  state = { error: false };
  static getDerivedStateFromError() {
    return { error: true };
  }
  componentDidUpdate(previous: { route: Route }) {
    if (previous.route !== this.props.route && this.state.error)
      this.setState({ error: false });
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Dashboard rendering failed", error, info.componentStack);
  }
  render() {
    return this.state.error ? (
      <Panel title="This view could not be displayed">
        <p>Refresh to retry loading the interface.</p>
        <Button onClick={() => location.reload()}>Refresh</Button>
      </Panel>
    ) : (
      this.props.children
    );
  }
}
export function App() {
  const { route, clusters, context, selectCluster, action, snapshot } =
    useDashboardContext();
  useEffect(() => {
    document.title = `${routes[route]} — Spark Cluster`;
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [route]);
  const views: Record<Route, ReactNode> = {
    overview: <Overview />,
    setup: <Setup />,
    api: <Api />,
    models: <Models />,
    runtimes: <Runtimes />,
    launch: <Launch />,
    logs: <Logs />,
    topology: <Topology />,
    cache: <Cache />,
    memory: <Memory />,
  };
  return (
    <ModelDetailsProvider>
      <div className="shell">
        <aside className="sidebar">
          <a className="brand" href="#overview">
            <Layers size={28} />
            <span>
              SPARK<small>CLUSTER CONTROL</small>
            </span>
          </a>
          <nav aria-label="Main navigation">
            {Object.entries(routes).map(([key, label]) => {
              const Icon = icons[key as Route];
              return (
                <a
                  key={key}
                  href={`#${key}`}
                  aria-current={route === key ? "page" : undefined}
                  className={route === key ? "active" : ""}
                >
                  <Icon size={17} />
                  {label}
                </a>
              );
            })}
          </nav>
          <div className="sidebar-footer">
            <i className="status-dot" />
            Connected infrastructure
          </div>
        </aside>
        <main>
          <header className="topbar">
            <div>
              <p className="eyebrow">SPARK CLUSTER</p>
              <h1>{routes[route]}</h1>
            </div>
            <div className="topbar-controls">
              <Field label="Compute pool">
                <Select
                  aria-label="Compute pool"
                  value={context}
                  onChange={(e) => void selectCluster(e.target.value)}
                  disabled={action.busy}
                >
                  <option value="all">All compute pools</option>
                  {clusters?.clusters.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ·{" "}
                      {c.kind === "local-docker"
                        ? `${c.gpu_count} GPUs`
                        : "remote"}
                    </option>
                  ))}
                </Select>
              </Field>
              <span className="note stream-status" title={snapshot?.timestamp}>
                {snapshot ? "● telemetry" : "○ connecting"}
              </span>
            </div>
          </header>
          <div className="mobile-route">
            <Select
              aria-label="Page"
              value={route}
              onChange={(e) => {
                location.hash = `#${e.target.value}`;
              }}
            >
              {Object.entries(routes).map(([k, l]) => (
                <option key={k} value={k}>
                  {l}
                </option>
              ))}
            </Select>
          </div>
          <Message error={action.error} />
          <div className="page">
            <ErrorBoundary route={route}>{views[route]}</ErrorBoundary>
          </div>
        </main>
      </div>
    </ModelDetailsProvider>
  );
}
function Api() {
  const { pool, poolError } = useDashboardContext();
  const endpoint = `${location.origin}/v1`;
  const available = pool?.clusters.filter((c) => c.available) || [];
  return (
    <>
      <Panel
        title="OpenAI pool endpoint"
        actions={
          <Badge good={!!available.length}>
            {available.length ? "● pool ready" : "○ pool unavailable"}
          </Badge>
        }
      >
        <Message error={poolError} />
        <p className="note">
          {available.length} clusters ·{" "}
          {pool?.models.filter((m) => m.loadable).length || 0} models ·
          automatic loading
        </p>
        <div className="endpoint">
          <code>{endpoint}</code>
          <Copy value={endpoint} />
        </div>
        <details>
          <summary>Routes</summary>
          {["models", "chat/completions", "embeddings"].map((path) => (
            <p key={path}>
              <code>
                {endpoint}/{path}
              </code>
            </p>
          ))}
          <p className="note">Any /v1 route · select with model</p>
        </details>
      </Panel>
      <Panel title="MCP endpoint">
        <p className="note">AI clients · admin token for changes</p>
        <div className="endpoint">
          <code>{location.origin}/mcp</code>
          <Copy value={`${location.origin}/mcp`} />
        </div>
      </Panel>
    </>
  );
}
