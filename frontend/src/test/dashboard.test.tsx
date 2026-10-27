import { beforeEach, describe, it, expect, vi } from "vitest";
import {
  render,
  screen,
  waitFor,
  within,
  fireEvent,
  act,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { App } from "@/App";
import { DashboardProvider, routes, type Route } from "@/lib/context";
import { PerformanceProvider } from "@/components/performance";
import { AuthDialog } from "@/components/auth";
import { capacityContext } from "@/pages/launch";
import { memorySegments } from "@/pages/resources";
import { physicalHosts } from "@/pages/overview";
import {
  baseline,
  pool,
  model,
  cluster,
  snapshot,
  catalog,
  fit,
  activity,
  host,
  library,
  guide,
} from "./fixtures";
import type { PoolState } from "@/lib/types";
class Socket {
  static instances: Socket[] = [];
  static OPEN = 1;
  readyState = 1;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  constructor(public url: string) {
    Socket.instances.push(this);
  }
  close() {
    this.closed = true;
  }
  send() {}
  message(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }
}
let requests: { path: string; method: string; body: Record<string, unknown> }[];
let override: (
  path: string,
  method: string,
  body: Record<string, unknown>,
) => unknown;
function mount(route: Route) {
  history.replaceState(null, "", `#${route}`);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 1000, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={client}>
      <DashboardProvider>
        <PerformanceProvider>
          <App />
          <AuthDialog />
        </PerformanceProvider>
      </DashboardProvider>
    </QueryClientProvider>,
  );
}
beforeEach(() => {
  requests = [];
  Socket.instances = [];
  vi.stubGlobal("WebSocket", Socket);
  override = () => undefined;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const path = String(input);
      const method = options?.method || "GET";
      const body = options?.body ? JSON.parse(String(options.body)) : {};
      requests.push({ path, method, body });
      const value = await override(path, method, body);
      return value instanceof Response
        ? value
        : new Response(JSON.stringify(value ?? baseline(path)), {
            status: 200,
          });
    }),
  );
});
describe("React dashboard parity", () => {
  it.each(Object.keys(routes) as Route[])(
    "renders the %s route without a page crash",
    async (route) => {
      mount(route);
      await screen.findByRole("heading", { name: routes[route], level: 1 });
      await waitFor(() =>
        expect(requests.some((r) => r.path === "/api/v1/pool")).toBe(true),
      );
      expect(screen.queryByText("This view could not be displayed")).toBeNull();
    },
  );
  it("uses automatic placement across all pools and preserves an explicit pool when selected", async () => {
    const user = userEvent.setup();
    mount("launch");
    await user.click(
      await screen.findByRole("button", { name: /^Launch model$/ }),
    );
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/pool/load")?.body,
      ).toEqual({ model_id: model.id, cluster_id: null }),
    );
    await user.selectOptions(screen.getByLabelText("Compute pool"), cluster.id);
    await user.click(screen.getByRole("button", { name: /^Launch model$/ }));
    await waitFor(() =>
      expect(
        requests.filter((r) => r.path === "/api/v1/pool/load").at(-1)?.body
          .cluster_id,
      ).toBe(cluster.id),
    );
  });
  it("rolls back a failed pool selection without overwriting browser context", async () => {
    const user = userEvent.setup();
    override = (path, method) =>
      path === "/api/v1/clusters"
        ? {
            clusters: [
              cluster,
              { ...cluster, id: "pool-two", name: "Other pool" },
            ],
            active_cluster_id: cluster.id,
          }
        : path === "/api/v1/clusters/pool-two/select" && method === "POST"
          ? new Response(JSON.stringify({ detail: "Controller unreachable" }), {
              status: 502,
            })
          : undefined;
    mount("launch");
    await screen.findByRole("button", { name: /^Launch model$/ });
    await user.selectOptions(screen.getByLabelText("Compute pool"), "pool-two");
    await screen.findByText("Controller unreachable");
    expect(
      (screen.getByLabelText("Compute pool") as HTMLSelectElement).value,
    ).toBe("all");
    expect(
      window.localStorage.getItem("spark-cluster.cluster-context"),
    ).toBeNull();
  });
  it("saves model residency and advanced RAM offload settings while retaining the original launch configuration", async () => {
    const user = userEvent.setup();
    mount("launch");
    await user.click(
      await screen.findByRole("button", { name: `Settings for ${model.id}` }),
    );
    const dialog = screen.getByRole("dialog");
    await user.click(within(dialog).getByLabelText("Keep loaded"));
    await user.click(within(dialog).getByText("Launch settings"));
    fireEvent.change(
      within(dialog).getByLabelText("RAM weight offload (GiB per GPU)"),
      { target: { value: "2" } },
    );
    await user.click(within(dialog).getByText("Save settings"));
    await waitFor(() =>
      expect(requests.some((r) => r.method === "PUT")).toBe(true),
    );
    const saved = requests.find((r) => r.method === "PUT")!.body;
    expect(saved.keep_loaded).toBe(true);
    expect(saved.last_settings).toMatchObject({
      model_id: model.id,
      cpu_offload_gib: 2,
      context_length: 8192,
      cuda_graphs: true,
      chunked_prefill: true,
    });
  });
  it("requires a fresh fit before downloading and sends the exact selected parameters", async () => {
    const user = userEvent.setup();
    mount("models");
    await user.click(
      await screen.findByRole("button", { name: /acme.*chat/i }),
    );
    await screen.findByText("README · README.md");
    const dialog = screen.getByRole("dialog");
    const download = within(dialog).getByRole("button", {
      name: "Download for later",
    });
    expect((download as HTMLButtonElement).disabled).toBe(true);
    await user.click(within(dialog).getByRole("button", { name: "Check fit" }));
    await screen.findByText("Estimated fit in cluster memory");
    await user.click(
      within(dialog).getByRole("button", { name: "Download model" }),
    );
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/models/download")?.body,
      ).toEqual({
        repo_id: "acme/chat",
        revision: "main",
        bits_per_weight: 4,
        context_length: 32768,
        concurrent_sequences: 1,
        kv_cache_dtype: "auto",
        allow_oversized: false,
      }),
    );
  });
  it("discards a late fit response after parameters change", async () => {
    const user = userEvent.setup();
    let resolve: (data: unknown) => void = () => {};
    override = (path) =>
      path === "/api/v1/models/fit"
        ? new Promise((r) => {
            resolve = r;
          })
        : undefined;
    mount("models");
    await user.click(
      await screen.findByRole("button", { name: /acme.*chat/i }),
    );
    await screen.findByText("README · README.md");
    await user.click(screen.getByRole("button", { name: "Check fit" }));
    await user.click(screen.getByText("Download & capacity options"));
    await user.selectOptions(screen.getByLabelText("Context length"), "8192");
    await act(async () => resolve(fit));
    await waitFor(() =>
      expect(
        (screen.getByRole("button", { name: "Check fit" }) as HTMLButtonElement)
          .disabled,
      ).toBe(false),
    );
    expect(screen.queryByText("Estimated fit in cluster memory")).toBeNull();
    expect(
      (
        screen.getByRole("button", {
          name: "Download for later",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
  it("keeps download controls behind a review dialog, including resolving and transfer preparation phases", async () => {
    const user = userEvent.setup();
    override = (path) =>
      path === "/api/v1/downloads"
        ? [
            {
              id: "job-1",
              repo_id: model.id,
              revision: "main",
              status: "preparing-transfer",
              progress: 100,
              transfer_progress: 0,
              downloaded_bytes: 1024,
              transferred_bytes: 0,
              total_bytes: 1024,
              message: "Preparing LAN transfer",
            },
          ]
        : undefined;
    mount("models");
    await user.click(await screen.findByRole("button", { name: /^Cancel$/ }));
    expect(requests.some((r) => r.path.endsWith("/cancel"))).toBe(false);
    await screen.findByRole("heading", {
      name: "Cancel and remove this download?",
    });
    await user.click(screen.getByRole("button", { name: "Confirm cancel" }));
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/downloads/job-1/cancel")
          ?.method,
      ).toBe("POST"),
    );
  });
  it("opens filters only on request and applies bounded catalog fit batches", async () => {
    const user = userEvent.setup();
    override = (path, method, body) =>
      path === "/api/v1/catalog/fit"
        ? {
            results: [
              {
                repo_id: catalog.id,
                compatible: true,
                minimum_nodes: 1,
                memory_fits: true,
                launch_supported: true,
                placement_strategy: "full-memory",
              },
            ],
            cache: {},
          }
        : undefined;
    mount("models");
    await screen.findByRole("button", { name: /acme.*chat/i });
    expect(screen.queryByLabelText("Pool capacity")).toBeNull();
    await user.click(screen.getByRole("button", { name: /^Filters$/ }));
    await user.selectOptions(screen.getByLabelText("Pool capacity"), "2");
    await user.click(screen.getByRole("button", { name: "Show models" }));
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/catalog/fit")?.body,
      ).toMatchObject({
        models: [{ id: "acme/chat", commit: null }],
        spark_count: 2,
        context_length: 32768,
        concurrent_sequences: 1,
        kv_cache_dtype: "auto",
      }),
    );
  });
  it("deduplicates physical host RAM and bounds unknown KV allocation", () => {
    expect(physicalHosts(snapshot.nodes)).toHaveLength(1);
    const parts = memorySegments({
      total_gib: 8,
      used_gib: 7,
      model_gib: 6,
      kv_cache_gib: 5,
      kv_cache_source: "unavailable",
    });
    expect(parts).toEqual([
      ["Model", 6],
      ["KV cache", 0],
      ["Used", 1],
      ["Free", 1],
    ]);
    expect(parts.reduce((s, [, v]) => s + v, 0)).toBe(8);
  });
  it("does not treat capacity on another pool as a fit for the selected pool", () => {
    expect(capacityContext(model, "all").fits).toBe(true);
    expect(capacityContext(model, "pool-two").fits).toBe(false);
    expect(
      capacityContext(
        {
          ...model,
          cluster_capacity: [
            { ...model.cluster_capacity[0], fits: false, can_load: true },
          ],
        },
        cluster.id,
      ).fits,
    ).toBe(true);
  });
  it("keeps native Mac registration behind readiness checks and explicit installation approval", async () => {
    const user = userEvent.setup();
    const candidate = {
      address: "192.168.50.20",
      hostname: "Test Mac",
      source: "manual",
      setup_mode: "native-macos",
      machine_kind: "mac-apple-silicon",
      username: "tester",
      fingerprint: "SHA256:test",
    };
    override = (path, method) =>
      path === "/api/v1/setup/discover"
        ? { nodes: [candidate], notes: ["One machine found"] }
        : path === "/api/v1/setup/bootstrap" && method === "POST"
          ? {
              id: "bootstrap-1",
              status: "ready",
              username: "tester",
              fingerprint: "SHA256:test",
            }
          : path === "/api/v1/setup/check"
            ? {
                nodes: [candidate],
                checks: [
                  {
                    node: candidate.address,
                    code: "metal",
                    status: "pass",
                    message: "Metal verified",
                    remedy: "",
                    automatic: false,
                  },
                ],
                topology: null,
              }
            : undefined;
    mount("setup");
    await user.click(screen.getByRole("button", { name: "Add compute pool" }));
    await user.click(
      screen.getByRole("button", { name: /^Discover machines$/ }),
    );
    await user.click(await screen.findByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Choose machines →" }));
    await screen.findByText("✓ tester · SSH verified");
    await user.click(screen.getByRole("button", { name: "Check readiness →" }));
    expect(
      (
        screen.getByRole("button", {
          name: "Register Mac →",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Run checks" }));
    await screen.findByText("✓ Metal verified");
    expect(
      (
        screen.getByRole("button", {
          name: "Register Mac →",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    await user.click(
      screen.getByRole("checkbox", {
        name: "Register this Mac as a trusted native runtime host.",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Register Mac →" }));
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/setup/jobs")?.body,
      ).toMatchObject({
        topology: "single",
        nodes: [
          {
            address: candidate.address,
            username: "tester",
            role: "head",
            fingerprint: "SHA256:test",
            fabric_address: null,
          },
        ],
      }),
    );
  });
  it("auto-selects a sole native host and installs a verified compatible package on that host", async () => {
    const user = userEvent.setup();
    override = (path) =>
      path === "/api/v1/clusters"
        ? { clusters: [], active_cluster_id: null }
        : path === "/api/v1/runtime-library"
          ? {
              ...library,
              hosts: [
                {
                  id: "host-abc123abc123",
                  name: "Test Mac",
                  address: "192.168.50.20",
                  username: "tester",
                  platform: "darwin/arm64",
                },
              ],
              artifacts: [
                {
                  id: "artifact-1",
                  runtime_id: "llama-cpp",
                  version: "b123",
                  platform: "darwin/arm64",
                  archive_bytes: 1024,
                  artifact_kind: "native",
                },
              ],
            }
          : undefined;
    mount("runtimes");
    await screen.findByText(
      "Native host ready. Select a verified native package for the registered host.",
    );
    await user.click(screen.getByText(/Cached packages/));
    await user.click(screen.getByRole("button", { name: "Install package" }));
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/runtime-library/installs")
          ?.body,
      ).toEqual({ artifact_id: "artifact-1", host_id: "host-abc123abc123" }),
    );
  });
  it("reviews runtime preparation before creating its guided workflow", async () => {
    const user = userEvent.setup();
    override = (path) =>
      path.startsWith("/api/v1/runtime-management/engines")
        ? {
            cluster_id: cluster.id,
            platform: "linux/amd64",
            engines: [
              {
                id: "vllm",
                name: "vLLM",
                capability: "managed_launch",
                image: "image:v1",
                ready: false,
                state: "needs_preparation",
                nodes: [],
                nodes_total: 1,
                nodes_ready: 0,
                message: "Runtime needs preparation",
                active_job_id: null,
                prepare_supported: false,
              },
            ],
          }
        : undefined;
    mount("runtimes");
    await user.click(
      await screen.findByRole("button", { name: "Set up vLLM" }),
    );
    await screen.findByRole("dialog");
    expect(
      requests.some((r) => r.path === "/api/v1/runtime-management/workflows"),
    ).toBe(false);
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Set up vLLM",
      }),
    );
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/runtime-management/workflows")
          ?.body,
      ).toEqual({
        cluster_id: cluster.id,
        engine_id: "vllm",
        model_id: null,
        launch: null,
      }),
    );
  });
  it("pauses log rendering, retains incoming lines and closes sockets on unmount", async () => {
    const user = userEvent.setup();
    const view = mount("logs");
    await screen.findByText("Logs will appear when a runtime starts.");
    const socket = Socket.instances.find((s) =>
      s.url.includes("/logs/stream"),
    )!;
    const log = {
      timestamp: new Date().toISOString(),
      pod: "runtime-1",
      container: "vllm",
      level: "info",
      message: "first line",
    };
    await act(async () => socket.message(log));
    await screen.findByText("first line");
    await user.click(screen.getByRole("button", { name: /^Pause$/ }));
    await act(async () => socket.message({ ...log, message: "second line" }));
    expect(screen.queryByText("second line")).toBeNull();
    await user.click(screen.getByRole("button", { name: /^Resume$/ }));
    await screen.findByText("second line");
    view.unmount();
    expect(Socket.instances.every((s) => s.closed)).toBe(true);
  });
  it("opens resource history and returns focus after Escape", async () => {
    const user = userEvent.setup();
    mount("overview");
    const trigger = await screen.findByRole("button", {
      name: "View gpu history for spark-1",
    });
    await user.click(trigger);
    await screen.findByRole("dialog");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(trigger));
  });
});
