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
function mount(route: Route, tab?: string) {
  history.replaceState(null, "", `#${route}${tab ? `?tab=${tab}` : ""}`);
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
  it("lets the user choose disk files to remove and explicitly retry a rejected launch", async () => {
    const user = userEvent.setup();
    let deleted = false;
    let attempts = 0;
    const locationId = cluster.id + ":disk";
    const assetId = cluster.id + ":old-copy";
    override = (path, method, body) => {
      if (path === "/api/v1/storage/delete") {
        deleted = true;
        return { deleted: [assetId], errors: [] };
      }
      if (path === "/api/v1/storage")
        return {
          locations: [
            {
              id: locationId,
              cluster_id: cluster.id,
              cluster_name: cluster.name,
              name: "Model disk",
              paths: ["/models"],
              free_bytes: deleted ? 200 : 10,
              total_bytes: 1000,
              scope: "node",
              node_ids: ["node"],
              node_names: ["Node"],
              kind: "filesystem",
            },
          ],
          assets: deleted
            ? []
            : [
                {
                  id: assetId,
                  location_id: locationId,
                  name: "owner/old",
                  model_id: "owner/old",
                  path: "/models/owner/old",
                  kind: "weights",
                  size_bytes: 200,
                  reclaimable_bytes: 200,
                  blockers: [],
                  recipe_refs: [{ id: "owner/recipe", version: "1.0.0" }],
                },
              ],
          errors: [],
        };
      if (path === "/api/v1/pool/load" && method === "POST") {
        attempts += 1;
        if (attempts === 1)
          return new Response(
            JSON.stringify({
              detail: {
                message: "Not enough disk space",
                code: "insufficient_storage",
                storage: {
                  requirements: [
                    {
                      location_id: locationId,
                      name: cluster.name,
                      required_bytes: 100,
                      available_bytes: 10,
                      missing_bytes: 90,
                    },
                  ],
                  excluded_model_ids: [model.id],
                },
              },
            }),
            { status: 507 },
          );
        return { id: "new-load", status: "queued" };
      }
      return undefined;
    };
    mount("launch");
    await user.click(
      await screen.findByRole("button", { name: /^Launch model$/ }),
    );
    await screen.findByRole("heading", { name: "Free disk space to continue" });
    expect(attempts).toBe(1);
    await user.click(
      await screen.findByRole("checkbox", {
        name: `Delete owner/old from ${cluster.name}`,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: "Delete selected files permanently" }),
    );
    await screen.findByText(
      "Selected files removed. Storage has been refreshed.",
    );
    expect(
      requests.find((r) => r.path === "/api/v1/storage/delete")?.body,
    ).toEqual({ asset_ids: [assetId] });
    expect(attempts).toBe(1);
    await user.click(screen.getByRole("button", { name: "Retry operation" }));
    await waitFor(() => expect(attempts).toBe(2));
  });

  it("installs the selected immutable recipe release on the selected pool", async () => {
    const user = userEvent.setup();
    const release = (version: string, hash: string) => ({
      id: hash,
      origin: { kind: "bundled" },
      blockers: [],
      recipe: {
        id: "publisher/qwen",
        version,
        title: "Qwen recipe",
        description: "Pinned setup",
        license: "Apache-2.0",
        validation: "source-reviewed",
        notes: [],
        source: {
          repository: "publisher/recipes",
          commit: "a".repeat(40),
          path: "recipe.json",
        },
        model: {
          repository: "nvidia/Qwen",
          revision: "b".repeat(40),
          license: "model-license",
        },
        runtime: {
          image: "docker.io/test/image@sha256:" + "c".repeat(64),
          unconfined: false,
        },
        requirements: {
          nodes: 1,
          memory_gib_per_node: 115,
          disk_gib_per_node: 250,
        },
      },
    });
    override = (path) =>
      path.startsWith("/api/v1/recipes")
        ? {
            recipes: [
              release("1.10.0", "a".repeat(64)),
              release("1.9.0", "b".repeat(64)),
            ],
            jobs: [],
            discovery: { checked_at: null, errors: [] },
          }
        : undefined;
    mount("models");
    await user.click(screen.getByRole("tab", { name: "Recipes", exact: true }));
    const select = await screen.findByLabelText("Version for Qwen recipe");
    await user.selectOptions(select, "b".repeat(64));
    expect(
      (screen.getByLabelText("Version for Qwen recipe") as HTMLSelectElement)
        .value,
    ).toBe("b".repeat(64));
    await user.click(screen.getByRole("button", { name: "Install & launch" }));
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/recipes/install")?.body,
      ).toEqual({ recipe_hash: "b".repeat(64), cluster_id: cluster.id }),
    );
  });

  it("pages a large recipe catalog and forwards search and workload filters", async () => {
    const user = userEvent.setup();
    override = (path) => {
      if (!path.startsWith("/api/v1/recipes")) return undefined;
      const params = new URL(path, "http://test").searchParams;
      const offset = Number(params.get("offset") || 0);
      const refs = params.get("category") === "text-to-image";
      return {
        recipes: refs
          ? []
          : Array.from({ length: 24 }, (_, index) => ({
              id: `hash-${offset + index}`,
              origin: {
                kind: "generated",
                task: "text-generation",
                downloads: 1000,
                likes: 5,
              },
              blockers: ["Requires Spark"],
              recipe: {
                id: `popular-models/model-${offset + index}`,
                version: "1.0.0",
                title: `Serving model ${offset + index}`,
                description: "Pinned template",
                validation: "generated-template",
                license: "Apache-2.0",
                notes: [],
                source: {
                  repository: "vllm-project/vllm",
                  commit: "a".repeat(40),
                  path: "registry.py",
                },
                model: {
                  repository: `models/model-${offset + index}`,
                  revision: "b".repeat(40),
                  license: "mit",
                },
                runtime: {
                  image: "docker.io/vllm/vllm-openai@sha256:" + "c".repeat(64),
                  unconfined: false,
                },
                requirements: {
                  nodes: 1,
                  memory_gib_per_node: 20,
                  disk_gib_per_node: 40,
                },
              },
            })),
        references: refs
          ? [
              {
                id: "ref",
                model_id: "images/Flux",
                revision: "d".repeat(40),
                task: "text-to-image",
                library: "diffusers",
                license: "apache-2.0",
                downloads: 50000,
                likes: 100,
                model_card: "https://huggingface.co/images/Flux",
                blocker: "Dedicated runtime adapter required",
              },
            ]
          : [],
        total: refs ? 1 : 1179,
        catalog_total: 1181,
        generated_total: 1179,
        has_more: !refs,
        catalog: {
          reference_count: 160,
          snapshots: [
            {
              captured_at: "2026-11-02T08:00:00Z",
              ranking: "Popularity snapshot",
            },
          ],
          upstream_libraries: [],
        },
        jobs: [],
        discovery: { checked_at: null, errors: [] },
      };
    };
    mount("models");
    await user.click(screen.getByRole("tab", { name: "Recipes", exact: true }));
    await screen.findByRole("heading", { name: "Serving model 0" });
    expect(screen.getAllByText("Generated template · untested")).toHaveLength(
      24,
    );
    expect(
      screen.queryByRole("heading", { name: "Serving model 24" }),
    ).toBeNull();
    await user.click(screen.getByRole("button", { name: "Next recipes" }));
    await screen.findByRole("heading", { name: "Serving model 24" });
    expect(requests.some((r) => r.path.includes("offset=24"))).toBe(true);
    await user.type(screen.getByLabelText("Search recipes"), "Qwen");
    await waitFor(() =>
      expect(
        requests.some(
          (r) => r.path.includes("q=Qwen") && r.path.includes("offset=0"),
        ),
      ).toBe(true),
    );
    await user.selectOptions(
      screen.getByLabelText("Recipe workload"),
      "text-to-image",
    );
    await screen.findByRole("heading", { name: "images/Flux" });
    expect(
      (
        screen.getByRole("button", {
          name: "Runtime adapter required",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      screen.queryByRole("button", { name: "Install template & launch" }),
    ).toBeNull();
  });

  it("separates the model library into lazy-loaded tabs with keyboard navigation", async () => {
    const user = userEvent.setup();
    mount("models");
    await screen.findByRole("heading", { name: "Stored on disk" });
    expect(
      screen.queryByRole("heading", { name: "Recipes", exact: true }),
    ).toBeNull();
    expect(
      screen.queryByRole("heading", { name: "Discover models" }),
    ).toBeNull();
    expect(requests.some((r) => r.path.startsWith("/api/v1/recipes"))).toBe(
      false,
    );
    expect(
      requests.some((r) => r.path.startsWith("/api/v1/catalog/models?")),
    ).toBe(false);
    const downloaded = screen.getByRole("tab", { name: "Downloaded models" });
    downloaded.focus();
    await user.keyboard("{ArrowRight}");
    await screen.findByRole("heading", { name: "Recipes", exact: true });
    expect(
      screen
        .getByRole("tab", { name: "Recipes", exact: true })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(
      screen.queryByRole("heading", { name: "Stored on disk" }),
    ).toBeNull();
    expect(location.hash).toBe("#models?tab=recipes");
    await user.keyboard("{End}");
    await screen.findByRole("heading", { name: /Download history/ });
    expect(
      screen.queryByRole("heading", { name: "Recipes", exact: true }),
    ).toBeNull();
    await user.keyboard("{Home}");
    await screen.findByRole("heading", { name: "Stored on disk" });
  });

  it("opens recipe deep links and follows tab navigation from browser history", async () => {
    mount("models", "recipes");
    await screen.findByRole("heading", { name: "Recipes", exact: true });
    expect(
      screen.getByRole("heading", { name: "Model library", level: 1 }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("heading", { name: "Stored on disk" }),
    ).toBeNull();
    await act(async () => {
      history.replaceState(null, "", "#models?tab=models");
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });
    await screen.findByRole("heading", { name: "Discover models" });
    expect(
      screen.queryByRole("heading", { name: "Recipes", exact: true }),
    ).toBeNull();
  });

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
  it("checks advanced launch compatibility with the selected GPU and offload placement", async () => {
    const user = userEvent.setup();
    mount("launch");
    await user.click(await screen.findByText("Advanced launch & tuning"));
    await user.selectOptions(
      await screen.findByLabelText("Downloaded model"),
      model.id,
    );
    fireEvent.change(screen.getByLabelText("Layers on GPU"), {
      target: { value: "8" },
    });
    fireEvent.change(screen.getByLabelText("tensor parallel"), {
      target: { value: "2" },
    });
    await user.click(
      screen.getByRole("button", { name: "Check launch compatibility" }),
    );
    await waitFor(() =>
      expect(
        requests.find((r) => r.path === "/api/v1/models/fit-local")?.body,
      ).toMatchObject({
        repo_id: model.id,
        gpu_layers: 8,
        cpu_offload_gib: 0,
        offload_ranks: 2,
      }),
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
    await user.click(screen.getByRole("tab", { name: "Models", exact: true }));
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
  it("copies an existing controller GGUF subset to the selected remote pool", async () => {
    const user = userEvent.setup();
    override = (path) =>
      path === "/api/v1/clusters"
        ? {
            clusters: [{ ...cluster, kind: "remote-bridge" }],
            active_cluster_id: cluster.id,
          }
        : path === "/api/v1/controller/models"
          ? [{ id: "acme/chat" }]
          : path.startsWith("/api/v1/catalog/models/acme/chat")
            ? { ...catalog, files: [{ name: "model-Q2.gguf", size: 100 }] }
            : undefined;
    mount("models");
    await user.click(screen.getByRole("tab", { name: "Models", exact: true }));
    await user.click(
      await screen.findByRole("button", { name: /acme.*chat/i }),
    );
    const dialog = screen.getByRole("dialog");
    await within(dialog).findByText(/Already on controller storage/);
    await user.click(within(dialog).getByRole("button", { name: "Check fit" }));
    await screen.findByText("Estimated fit in cluster memory");
    await user.click(
      within(dialog).getByRole("button", { name: "Copy from controller" }),
    );
    await waitFor(() =>
      expect(
        requests.find((request) => request.path === "/api/v1/models/download")
          ?.body,
      ).toMatchObject({
        repo_id: "acme/chat",
        smallest_gguf_transfer_only: true,
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
    await user.click(screen.getByRole("tab", { name: "Models", exact: true }));
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
    await user.click(
      screen.getByRole("tab", { name: "Downloads", exact: true }),
    );
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
    await user.click(screen.getByRole("tab", { name: "Models", exact: true }));
    const modelCard = await screen.findByRole("button", {
      name: /acme.*chat/i,
    });
    expect(modelCard.querySelector(".catalog-cover")).not.toBeNull();
    expect(
      modelCard.querySelector(".catalog-cover img")?.getAttribute("src"),
    ).toBe(catalog.thumbnail_url);
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
  it("offers administrator authentication for runtime packages even without a compute pool", async () => {
    const user = userEvent.setup();
    override = (path) =>
      path === "/api/v1/clusters"
        ? { clusters: [], active_cluster_id: null }
        : path === "/api/v1/runtime-library" ||
            path === "/api/v1/controller/verify"
          ? new Response(
              JSON.stringify({ detail: "Invalid model administration token" }),
              { status: 401 },
            )
          : undefined;
    mount("runtimes");
    const access = await screen.findByRole("button", {
      name: "Administrator access",
    });
    expect((access as HTMLButtonElement).disabled).toBe(false);
    await user.click(access);
    expect((await screen.findByRole("dialog")).textContent).toContain(
      "Controller token required",
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
