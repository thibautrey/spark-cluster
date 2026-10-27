import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { api, useApi, post, useAction } from "./api";
import { defaultLaunch } from "./launch-defaults";
import type { ClusterSnapshot } from "./schema";
import type { ClusterState, PoolState } from "./types";
export const routes = {
  setup: "Compute pools",
  overview: "Pool overview",
  api: "Inference API",
  models: "Model library",
  runtimes: "Runtime readiness",
  launch: "Launch a model",
  logs: "Runtime logs",
  topology: "Pool topology",
  cache: "KV cache",
  memory: "Memory",
};
export type Route = keyof typeof routes;
const routeFromHash = () =>
  location.hash.slice(1) in routes
    ? (location.hash.slice(1) as Route)
    : "overview";
export function useRoute() {
  const [route, setRoute] = useState<Route>(routeFromHash);
  useEffect(() => {
    const change = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  return route;
}
export function useSocket<T>(path: string, onData: (data: T) => void) {
  useEffect(() => {
    let socket: WebSocket;
    let timer: ReturnType<typeof setTimeout>;
    let stopped = false;
    function connect() {
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${path}`,
      );
      socket.onmessage = (e) => {
        try {
          onData(JSON.parse(e.data) as T);
        } catch {
          /* malformed frame is ignored */
        }
      };
      socket.onclose = () => {
        if (!stopped) timer = setTimeout(connect, 1500);
      };
    }
    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      socket.close();
    };
  }, [path, onData]);
}
function storedContext() {
  try {
    return localStorage.getItem("spark-cluster.cluster-context") || "all";
  } catch {
    return "all";
  }
}
function useDashboard() {
  const route = useRoute();
  const [launchSettings, setLaunchSettings] = useState(defaultLaunch);
  const clusters = useApi<ClusterState>("/api/v1/clusters", 10000);
  const pool = useApi<PoolState>("/api/v1/pool", 3000);
  const snapshot = useApi<ClusterSnapshot>("/api/v1/snapshot", 10000);
  const [live, setLive] = useState<ClusterSnapshot>();
  useSocket("/api/v1/stream", setLive);
  const [context, setContext] = useState(storedContext);
  const action = useAction();
  const selected =
    context === "all" ||
    !clusters.data ||
    clusters.data.clusters.some((c) => c.id === context)
      ? context
      : "all";
  async function selectCluster(id: string) {
    const result = await action.run(async () => {
      if (id !== "all" && id !== clusters.data?.active_cluster_id)
        await post(`/api/v1/clusters/${encodeURIComponent(id)}/select`);
      return id;
    });
    if (result !== undefined) {
      try {
        localStorage.setItem("spark-cluster.cluster-context", id);
      } catch {
        /* storage unavailable */
      }
      setContext(id);
      setLive(undefined);
      void snapshot.refetch();
    }
  }
  return {
    route,
    launchSettings,
    setLaunchSettings,
    clusters: clusters.data,
    pool: pool.data,
    poolError: pool.error,
    snapshot:
      live &&
      (!snapshot.data ||
        Date.parse(live.timestamp) >= Date.parse(snapshot.data.timestamp))
        ? live
        : snapshot.data,
    snapshotError: snapshot.error,
    context: selected,
    selectCluster,
    action,
    refresh: () =>
      Promise.all([clusters.refetch(), pool.refetch(), snapshot.refetch()]),
  };
}
const Context = createContext<ReturnType<typeof useDashboard> | null>(null);
export function DashboardProvider({ children }: { children: ReactNode }) {
  const state = useDashboard();
  return <Context.Provider value={state}>{children}</Context.Provider>;
}
export function useDashboardContext() {
  const state = useContext(Context);
  if (!state) throw new Error("Dashboard context missing");
  return state;
}
