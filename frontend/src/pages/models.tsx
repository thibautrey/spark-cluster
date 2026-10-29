import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { api, post, useAction, useApi } from "@/lib/api";
import { useDashboardContext } from "@/lib/context";
import type {
  CatalogPage,
  CatalogModel,
  CatalogDetail,
  CatalogFitBatchResponse,
  ModelFitPlan,
  DownloadJob,
  LocalModel,
} from "@/lib/schema";
import { bytes, number, repoPath, errorMessage } from "@/lib/utils";
import { renderModelReadme } from "@/lib/readme";
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
  Meter,
  Badge,
  Facts,
} from "@/components/common";
import { PerformanceButton } from "@/components/performance";
const DetailsContext = createContext<(id: string, revision?: string) => void>(
  () => {},
);
export const useModelDetails = () => useContext(DetailsContext);
export function ModelDetailsProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState<{ id: string; revision: string } | null>(
    null,
  );
  return (
    <DetailsContext.Provider
      value={(id, revision = "main") => setModel({ id, revision })}
    >
      {children}
      {model && (
        <ModelDetails
          key={`${model.id}-${model.revision}`}
          model={model}
          close={() => setModel(null)}
        />
      )}
    </DetailsContext.Provider>
  );
}
function ModelDetails({
  model,
  close,
}: {
  model: { id: string; revision: string };
  close: () => void;
}) {
  const [repo, setRepo] = useState(model.id);
  const [revision, setRevision] = useState(model.revision);
  const { clusters } = useDashboardContext();
  const target = clusters?.clusters.find(
    (item) => item.id === clusters.active_cluster_id,
  );
  const controllerModels = useApi<LocalModel[]>(
    target?.kind === "remote-bridge" ? "/api/v1/controller/models" : null,
  );
  const reusableCopy = controllerModels.data?.find(
    (item) => item.id === repo.trim(),
  );
  const [smallestGgufOnly, setSmallestGgufOnly] = useState(true);
  const detail = useApi<CatalogDetail>(
    repo
      ? `/api/v1/catalog/models/${repoPath(repo)}?revision=${encodeURIComponent(revision)}`
      : null,
  );
  const [bits, setBits] = useState(4);
  const [context, setContext] = useState(32768);
  const [concurrency, setConcurrency] = useState(1);
  const [kv, setKv] = useState("auto");
  const [fit, setFit] = useState<ModelFitPlan | null>(null);
  const action = useAction();
  const signature = JSON.stringify({
    repo_id: repo.trim(),
    revision: revision.trim() || "main",
    bits_per_weight: bits,
    context_length: context,
    concurrent_sequences: concurrency,
    kv_cache_dtype: kv,
  });
  const latest = useRef(signature);
  latest.current = signature;
  useEffect(() => setFit(null), [signature]);
  const md = detail.data;
  const readme = useMemo(
    () => (md ? renderModelReadme(md.readme || "", md) : ""),
    [md],
  );
  return (
    <Sheet
      open
      onOpenChange={(v) => {
        if (!v) close();
      }}
      title={repo ? repo.split("/").at(-1)! : "Add repository"}
    >
      <Message error={detail.error} />
      {detail.error && (
        <Button variant="outline" onClick={() => detail.refetch()}>
          Retry inspection
        </Button>
      )}
      {detail.isLoading && <p role="status">Inspecting repository files…</p>}
      {md && (
        <>
          <div className="split">
            <div className="model-identity">
              <Avatar model={md} />
              <div>
                <small>{md.author}</small>
                <h3>{md.id}</h3>
              </div>
            </div>
            <a
              href={`https://huggingface.co/${repoPath(md.id)}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Model card & license ↗
            </a>
          </div>
          <div className="actions">
            <Badge>{md.task}</Badge>
            <Badge>{md.library}</Badge>
            <span>
              {number(md.downloads)} downloads · {number(md.likes)} likes
            </span>
          </div>
          <Facts
            items={[
              [
                "Repository size",
                `${md.size_complete ? "" : "At least "}${bytes(md.repository_bytes)}`,
              ],
              ["Parameters", number(md.parameters)],
              ["Architecture", md.architecture || md.library || "Not reported"],
              ["License", md.license || "See model license"],
            ]}
          />
          <Recommendation model={md} />
          <PerformanceButton scope="model" entity={md.id}>
            Speed & cache history ↗
          </PerformanceButton>
          {md.gated && (
            <p className="note">
              License approval is required. Accept access on Hugging Face and
              configure the controller token.
            </p>
          )}
          <details>
            <summary>README · {md.readme_path || "README.md"}</summary>
            <article
              className="model-markdown"
              dangerouslySetInnerHTML={{ __html: readme }}
            />
            {md.readme_truncated && (
              <p className="note">
                README truncated to 512 KiB. Open the model card to read the
                complete file.
              </p>
            )}
          </details>
          <details>
            <summary>{md.files.length} repository files</summary>
            {md.files.map((f) => (
              <div key={f.name} className="split">
                <code>{f.name}</code>
                <small>{f.size == null ? "Unknown size" : bytes(f.size)}</small>
              </div>
            ))}
          </details>
        </>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!fit?.download_fits) return;
          void action.run(async () => {
            await post("/api/v1/models/download", {
              ...JSON.parse(signature),
              allow_oversized: false,
              smallest_gguf_transfer_only: Boolean(
                reusableCopy &&
                smallestGgufOnly &&
                md?.files.some((file) => file.name.endsWith(".gguf")),
              ),
            });
          }, "Download queued.");
        }}
      >
        {reusableCopy && (
          <p className="note">
            Already on controller storage. Files will be copied over the LAN to{" "}
            {target?.name}; no Hugging Face download is needed.
          </p>
        )}
        {reusableCopy &&
          md?.files.some((file) => file.name.endsWith(".gguf")) && (
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={smallestGgufOnly}
                onChange={(event) => setSmallestGgufOnly(event.target.checked)}
              />
              Copy only the smallest GGUF variant and its shared files
            </label>
          )}
        <div className="form-grid">
          <Field label="Repository ID">
            <Input
              value={repo}
              onChange={(e) => setRepo(e.target.value)}
              placeholder="owner/model"
              required
            />
          </Field>
          <Field label="Revision">
            <Input
              value={revision}
              onChange={(e) => setRevision(e.target.value)}
              required
            />
          </Field>
        </div>
        <details>
          <summary>Download & capacity options</summary>
          <div className="form-grid">
            <Field label="Weight precision">
              <Select value={bits} onChange={(e) => setBits(+e.target.value)}>
                {[2, 3, 4, 8, 16, 32].map((v) => (
                  <option key={v} value={v}>
                    {v} bits
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Context length">
              <Select
                value={context}
                onChange={(e) => setContext(+e.target.value)}
              >
                {[8192, 32768, 131072, 524288, 1048576].map((v) => (
                  <option key={v} value={v}>
                    {v / 1024}k
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Concurrent sequences">
              <Input
                type="number"
                min={1}
                max={256}
                value={concurrency}
                onChange={(e) => setConcurrency(+e.target.value)}
              />
            </Field>
            <Field label="KV cache dtype">
              <Select value={kv} onChange={(e) => setKv(e.target.value)}>
                <option value="auto">Automatic</option>
                <option value="fp8_e4m3">FP8 E4M3</option>
              </Select>
            </Field>
          </div>
        </details>
        <Message error={action.error}>{action.message}</Message>
        {fit && <FitResult fit={fit} />}
        <div className="actions">
          <Button
            type="button"
            variant="outline"
            disabled={action.busy || !repo.trim()}
            onClick={() => {
              const current = signature;
              void action.run(async () => {
                const result = await post<ModelFitPlan>(
                  "/api/v1/models/fit",
                  JSON.parse(current),
                );
                if (latest.current === current) setFit(result);
              });
            }}
          >
            Check fit
          </Button>
          <Button disabled={action.busy || !fit?.download_fits}>
            {reusableCopy
              ? "Copy from controller"
              : fit?.runtime_fits
                ? "Download model"
                : "Download for later"}
          </Button>
        </div>
      </form>
    </Sheet>
  );
}
export function FitResult({ fit }: { fit: ModelFitPlan }) {
  return (
    <div className={`fit ${fit.download_fits ? "" : "error"}`}>
      <b>
        {
          {
            "full-memory": "Estimated fit in cluster memory",
            "hybrid-ready": "Estimated hybrid placement",
            "engine-required": "Download fits · specialized engine required",
            "download-only": "Download fits · runtime plan needs validation",
            "insufficient-storage": "Not enough model storage",
          }[fit.fit_status]
        }
      </b>
      <p>
        {fit.detected_format} · {fit.engine_recommendation} ·{" "}
        {fit.minimum_nodes} accelerators minimum · {bytes(fit.repository_bytes)}{" "}
        repository
      </p>
      <details>
        <summary>Technical details</summary>
        <Facts
          items={[
            ["Resident weights", number(fit.resident_weights_gib, " GiB")],
            ["Offloaded weights", number(fit.offloaded_weights_gib, " GiB")],
            ["Resident KV", number(fit.resident_kv_cache_gib, " GiB")],
            ["CPU/NVMe KV tier", number(fit.offloaded_kv_cache_gib, " GiB")],
            ["Runtime overhead", number(fit.runtime_overhead_gib, " GiB")],
            [
              "Runnable artifacts",
              number(fit.runtime_storage_required_gib, " GiB"),
            ],
          ]}
        />
        <p>
          {fit.placement_strategy} · {number(fit.storage_available_gib, " GiB")}{" "}
          storage free · {number(fit.ssd_required_gib_per_node, " GiB")} local
          SSD per node
        </p>
        {[
          ["Required", fit.requirements],
          ["Tradeoffs", fit.tradeoffs],
          ["Planner notes", fit.notes],
        ].map(([label, items]) => (
          <div key={String(label)}>
            <b>{String(label)}</b>
            {(items as string[]).map((t) => (
              <p key={t}>{t}</p>
            ))}
          </div>
        ))}
      </details>
    </div>
  );
}
function Avatar({
  model,
}: {
  model: { id: string; author: string; icon_url?: string | null };
}) {
  const profile = `/api/v1/catalog/profiles/${encodeURIComponent(model.author)}/avatar`;
  const [src, setSrc] = useState(model.icon_url || profile);
  const [failed, setFailed] = useState(false);
  return (
    <span className="avatar">
      {failed ? (
        model.id.split("/").at(-1)?.slice(0, 2).toUpperCase()
      ) : (
        <img
          src={src}
          alt={`${model.author} avatar`}
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={() => {
            if (src !== profile) setSrc(profile);
            else setFailed(true);
          }}
        />
      )}
    </span>
  );
}
function Recommendation({ model }: { model: CatalogModel }) {
  const r = model.recommendation;
  return (
    <>
      {r && r.priority !== 0 && (
        <details>
          <summary>
            <Badge good={r.priority > 0}>{r.label}</Badge>
          </summary>
          <p>
            {r.runtime} · reported on {r.reported_nodes.join(", ")} Sparks ·
            reviewed {r.reviewed}
          </p>
          {[...r.reasons, ...r.caveats].map((t) => (
            <p key={t}>{t}</p>
          ))}
          {r.sources.map((url, i) => (
            <a key={url} href={url} target="_blank" rel="noopener noreferrer">
              Evidence {i + 1} ↗{" "}
            </a>
          ))}
          <p className="note">
            Discovery evidence does not guarantee fit, quality, speed or launch
            support.
          </p>
        </details>
      )}
      {model.spark && (model.spark.score || model.spark.caveats.length > 0) ? (
        <details>
          <summary>⚡ {model.spark.label}</summary>
          {[...model.spark.reasons, ...model.spark.caveats].map((t) => (
            <p key={t}>{t}</p>
          ))}
        </details>
      ) : null}
    </>
  );
}
const activeStatuses = new Set([
  "queued",
  "resolving",
  "downloading",
  "paused",
  "preparing-transfer",
  "distributing",
]);
export function Models() {
  const inspect = useModelDetails();
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [sort, setSort] = useState("spark");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [task, setTask] = useState("");
  const [library, setLibrary] = useState("");
  const [capacity, setCapacity] = useState("");
  const [context, setContext] = useState("32768");
  const [selection, setSelection] = useState("standard");
  const [access, setAccess] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 300);
    return () => clearTimeout(t);
  }, [query]);
  const catalog = useInfiniteQuery({
    queryKey: ["catalog", debounced, sort, task, library, access],
    initialPageParam: "",
    queryFn: ({ pageParam, signal }) =>
      api<CatalogPage>(
        `/api/v1/catalog/models?${new URLSearchParams({ query: debounced, sort, task, library, access, ...(pageParam ? { cursor: pageParam } : {}) })}`,
        { signal },
      ),
    getNextPageParam: (page) => page.cursor || undefined,
    retry: false,
  });
  const candidates = useMemo(
    () => [
      ...new Map(
        (catalog.data?.pages.flatMap((p) => p.models) || []).map((m) => [
          m.id,
          m,
        ]),
      ).values(),
    ],
    [catalog.data],
  );
  const fit = useQuery({
    queryKey: [
      "catalog-fits",
      candidates.map((m) => `${m.id}:${m.commit}`),
      capacity,
      context,
    ],
    enabled: !!capacity && !!candidates.length,
    queryFn: async ({ signal }) => {
      const results: CatalogFitBatchResponse["results"] = [];
      for (let i = 0; i < candidates.length; i += 48) {
        const data = await api<CatalogFitBatchResponse>("/api/v1/catalog/fit", {
          method: "POST",
          signal,
          body: {
            models: candidates
              .slice(i, i + 48)
              .map((m) => ({ id: m.id, commit: m.commit })),
            spark_count: +capacity,
            context_length: +context,
            concurrent_sequences: 1,
            kv_cache_dtype: "auto",
          },
        });
        results.push(...data.results);
      }
      return new Map(results.map((r) => [r.repo_id, r]));
    },
    retry: false,
  });
  const models = candidates
    .filter((m) => {
      const r = m.recommendation;
      return (
        (selection === "standard"
          ? !r || r.variant === "standard"
          : selection === "recipes"
            ? r?.priority === 2
            : selection === "publisher"
              ? r?.publisher !== "community"
              : true) &&
        (!capacity || fit.data?.get(m.id)?.compatible)
      );
    })
    .sort((a, b) =>
      capacity && sort === "spark"
        ? Number(fit.data?.get(b.id)?.launch_supported) -
            Number(fit.data?.get(a.id)?.launch_supported) ||
          Number(b.recommendation?.reported_nodes.includes(+capacity)) -
            Number(a.recommendation?.reported_nodes.includes(+capacity))
        : 0,
    );
  return (
    <>
      <Downloads />
      <Inventory />
      <Panel
        title="Discover models"
        actions={
          <Button variant="outline" onClick={() => inspect("")}>
            ＋ Repository
          </Button>
        }
      >
        <div className="toolbar">
          <Input
            aria-label="Search Hugging Face models"
            placeholder="Search models or paste a repository ID…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && query.trim().includes("/"))
                inspect(query.trim());
            }}
          />
          <Select
            aria-label="Sort models"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            {[
              ["spark", "Recommended"],
              ["trendingScore", "Trending"],
              ["downloads", "Most downloaded"],
              ["likes", "Most liked"],
              ["lastModified", "Recently updated"],
            ].map(([v, l]) => (
              <option value={v} key={v}>
                {l}
              </option>
            ))}
          </Select>
          <Button variant="outline" onClick={() => setFiltersOpen(true)}>
            Filters
          </Button>
        </div>
        <Message error={catalog.error || fit.error} />
        <p role="status" className="note">
          {catalog.isFetching
            ? "Searching Hugging Face…"
            : fit.isFetching
              ? "Calculating model capacity…"
              : `${models.length} models${capacity ? ` fit ${capacity} accelerators at ${+context / 1024}k context` : ""}`}
        </p>
        <div className="card-grid">
          {models.map((m) => (
            <button
              type="button"
              className="model-card text-left"
              key={m.id}
              onClick={() => inspect(m.id)}
            >
              <span className="catalog-cover" aria-hidden="true">
                <span className="catalog-cover-fallback">
                  {(m.id.split("/").at(-1) || m.author)
                    .replace(/[^A-Za-z0-9]+/g, " ")
                    .trim()
                    .split(/\s+/)
                    .slice(0, 2)
                    .map((part) => part[0])
                    .join("")
                    .toUpperCase() || "HF"}
                </span>
                <img
                  src={m.thumbnail_url}
                  alt=""
                  loading="lazy"
                  decoding="async"
                  referrerPolicy="no-referrer"
                  onError={(event) => {
                    event.currentTarget.hidden = true;
                  }}
                />
                {m.gated && <em>License required</em>}
              </span>
              <div className="model-identity">
                <Avatar model={m} />
                <small>{m.author}</small>
              </div>
              <h3>{m.id.split("/").at(-1)}</h3>
              <div className="actions">
                <Badge>{m.task}</Badge>
                <Badge>{m.library}</Badge>
              </div>
              {m.recommendation && m.recommendation.priority > 1 && (
                <p className="good-text">{m.recommendation.label}</p>
              )}
              {capacity && fit.data?.get(m.id) && (
                <p className="good-text">
                  ✓ {fit.data.get(m.id)!.minimum_nodes} accelerators ·{" "}
                  {fit.data.get(m.id)!.placement_strategy}
                </p>
              )}
              <footer>
                {number(m.parameters)} params · ↓ {number(m.downloads)} · ♡{" "}
                {number(m.likes)}
              </footer>
            </button>
          ))}
        </div>
        {!models.length && !catalog.isLoading && !fit.isFetching && (
          <Empty>No models match this view.</Empty>
        )}
        <div className="actions">
          {catalog.isError && (
            <Button onClick={() => catalog.refetch()}>Retry</Button>
          )}
          {catalog.hasNextPage && (
            <Button
              variant="outline"
              disabled={catalog.isFetchingNextPage}
              onClick={() => catalog.fetchNextPage()}
            >
              Load more models
            </Button>
          )}
        </div>
      </Panel>
      <Sheet open={filtersOpen} onOpenChange={setFiltersOpen} title="Filters">
        <div className="form-grid">
          {[
            {
              label: "Task",
              value: task,
              set: setTask,
              options: [
                ["", "All tasks"],
                ["text-generation", "Text generation"],
                ["image-text-to-text", "Vision & language"],
                ["feature-extraction", "Embeddings"],
                ["text-to-image", "Text to image"],
                ["automatic-speech-recognition", "Speech recognition"],
              ],
            },
            {
              label: "Format",
              value: library,
              set: setLibrary,
              options: [
                ["", "All formats"],
                ...["transformers", "safetensors", "gguf", "diffusers"].map(
                  (x) => [x, x],
                ),
              ],
            },
            {
              label: "Pool capacity",
              value: capacity,
              set: setCapacity,
              options: [
                ["", "Any capacity"],
                ...[1, 2, 3, 4, 5, 6, 8].map((x) => [
                  String(x),
                  `${x} accelerators`,
                ]),
              ],
            },
            {
              label: "Fit context",
              value: context,
              set: setContext,
              options: [8192, 32768, 131072, 524288, 1048576].map((x) => [
                String(x),
                `${x / 1024}k`,
              ]),
            },
            {
              label: "Model selection",
              value: selection,
              set: setSelection,
              options: [
                ["standard", "Standard checkpoints"],
                ["recipes", "Documented Spark recipes"],
                ["publisher", "Family publisher / NVIDIA"],
                ["all", "All variants"],
              ],
            },
            {
              label: "Access",
              value: access,
              set: setAccess,
              options: [
                ["", "All models"],
                ["open", "Open access"],
                ["gated", "License approval"],
              ],
            },
          ].map((f) => (
            <Field label={f.label} key={f.label}>
              <Select value={f.value} onChange={(e) => f.set(e.target.value)}>
                {f.options.map(([v, l]) => (
                  <option key={v} value={v}>
                    {l}
                  </option>
                ))}
              </Select>
            </Field>
          ))}
        </div>
        <div className="actions">
          <Button
            variant="outline"
            onClick={() => {
              setTask("");
              setLibrary("");
              setCapacity("");
              setContext("32768");
              setSelection("standard");
              setAccess("");
            }}
          >
            Reset
          </Button>
          <Button onClick={() => setFiltersOpen(false)}>Show models</Button>
        </div>
      </Sheet>
    </>
  );
}
function Inventory() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const models = useApi<LocalModel[]>(open ? "/api/v1/models" : null, 10000);
  const inspect = useModelDetails();
  const [deleting, setDeleting] = useState<string | null>(null);
  const action = useAction();
  return (
    <Panel>
      <details open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
        <summary>
          Downloaded models {models.data ? `· ${models.data.length}` : ""}
        </summary>
        <Input
          type="search"
          aria-label="Search downloaded models"
          placeholder="Search downloaded models…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Message error={models.error} />
        {models.data
          ?.filter((m) => m.id.toLowerCase().includes(query.toLowerCase()))
          .map((m) => (
            <article className="list-row" key={m.id}>
              <div>
                <Button
                  variant="link"
                  onClick={() => inspect(m.id, m.revision || undefined)}
                >
                  {m.id}
                </Button>
                <small>
                  {bytes(m.size_bytes)} · {m.revision} ·{" "}
                  {m.loaded ? "loaded" : "downloaded"}
                </small>
              </div>
              <div className="actions">
                <a href="#launch">Launch ↗</a>
                <Button variant="outline" onClick={() => setDeleting(m.id)}>
                  Delete
                </Button>
              </div>
            </article>
          ))}
        {models.data?.length === 0 && <Empty>No downloaded models.</Empty>}
      </details>
      <Sheet
        open={!!deleting}
        onOpenChange={(v) => {
          if (!v) setDeleting(null);
        }}
        title="Delete this model everywhere?"
        description="Controller model files and every managed cluster copy will be deleted. Stop the model first if it is running."
      >
        <p>{deleting}</p>
        <Message error={action.error} />
        <div className="actions">
          <Button
            variant="outline"
            disabled={action.busy}
            onClick={() => setDeleting(null)}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={action.busy}
            onClick={() =>
              void action.run(async () => {
                await api(
                  `/api/v1/model-management/models/${repoPath(deleting!)}`,
                  { method: "DELETE", interactive: true },
                );
                setDeleting(null);
              })
            }
          >
            Delete model everywhere
          </Button>
        </div>
      </Sheet>
    </Panel>
  );
}
function Downloads() {
  const downloads = useApi<DownloadJob[]>("/api/v1/downloads", 2000);
  const hf = useApi<{ configured: boolean; source: string }>(
    "/api/v1/controller/huggingface",
    10000,
  );
  const inspect = useModelDetails();
  const action = useAction();
  const [control, setControl] = useState<{
    job: DownloadJob;
    action: string;
  } | null>(null);
  const [hfOpen, setHfOpen] = useState(false);
  const [token, setToken] = useState("");
  const active =
    downloads.data?.filter((j) => activeStatuses.has(j.status)) || [];
  const history =
    downloads.data?.filter((j) => !activeStatuses.has(j.status)) || [];
  const samples = useRef(
    new Map<string, { time: number; bytes: number; rate: number }>(),
  );
  function rate(j: DownloadJob) {
    const value =
      j.status === "distributing" ? j.transferred_bytes : j.downloaded_bytes;
    const previous = samples.current.get(j.id);
    if (!previous) {
      samples.current.set(j.id, { time: Date.now(), bytes: value, rate: 0 });
      return 0;
    }
    if (value !== previous.bytes) {
      const r = Math.max(
        0,
        (value - previous.bytes) /
          Math.max(0.001, (Date.now() - previous.time) / 1000),
      );
      samples.current.set(j.id, { time: Date.now(), bytes: value, rate: r });
      return r;
    }
    return previous.rate;
  }
  return (
    <>
      <Message error={downloads.error} />
      {!!active.length && (
        <Panel
          title="Downloads"
          actions={
            <Button variant="outline" onClick={() => setHfOpen(true)}>
              {hf.data?.configured
                ? "Hugging Face token configured"
                : "Add Hugging Face token"}
            </Button>
          }
        >
          {active.map((j) => {
            const distribution = j.status === "distributing";
            const current = distribution
              ? j.transferred_bytes
              : j.downloaded_bytes;
            const speed = j.status === "paused" ? 0 : rate(j);
            return (
              <article className="download-card" key={j.id}>
                <div className="split">
                  <Button
                    variant="link"
                    onClick={() => inspect(j.repo_id, j.revision)}
                  >
                    {j.repo_id}
                  </Button>
                  <Badge>{j.status}</Badge>
                </div>
                <Meter
                  value={distribution ? j.transfer_progress : j.progress}
                  label={`Download ${j.repo_id}`}
                />
                <div className="split">
                  <small>
                    {bytes(current)} of {bytes(j.total_bytes)} ·{" "}
                    {number(
                      distribution ? j.transfer_progress : j.progress,
                      "%",
                    )}{" "}
                    {speed > 0
                      ? `· ${bytes(speed)}/s · ${Math.ceil((j.total_bytes - current) / speed / 60)} min remaining`
                      : ""}
                  </small>
                  <div className="actions">
                    <Button
                      variant="outline"
                      onClick={() =>
                        setControl({
                          job: j,
                          action: j.status === "paused" ? "resume" : "pause",
                        })
                      }
                    >
                      {j.status === "paused" ? "Resume" : "Pause"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setControl({ job: j, action: "cancel" })}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
                <p className="note">{j.message}</p>
              </article>
            );
          })}
        </Panel>
      )}
      <Panel>
        <details>
          <summary>Download history · {history.length}</summary>
          {history.map((j) => (
            <article className="list-row" key={j.id}>
              <div>
                <Button
                  variant="link"
                  onClick={() => inspect(j.repo_id, j.revision)}
                >
                  {j.repo_id}
                </Button>
                <small>
                  {j.message} · attempt {j.attempts} · {j.destination} ·{" "}
                  {j.target_cluster_name}
                </small>
              </div>
              <Badge>{j.status}</Badge>
              {j.status === "failed" && (
                <Button
                  variant="outline"
                  disabled={action.busy}
                  onClick={() =>
                    void action.run(() =>
                      post(`/api/v1/downloads/${j.id}/retry`),
                    )
                  }
                >
                  Retry
                </Button>
              )}
            </article>
          ))}
          {!history.length && (
            <Empty>Completed and failed transfers will appear here.</Empty>
          )}
        </details>
        <Message error={action.error}>{action.message}</Message>
      </Panel>
      <Sheet
        open={!!control}
        onOpenChange={(v) => {
          if (!v) setControl(null);
        }}
        title={
          control?.action === "cancel"
            ? "Cancel and remove this download?"
            : control?.action === "pause"
              ? "Pause this model download?"
              : "Resume this model download?"
        }
        description={
          control?.action === "cancel"
            ? "The transfer stops and partial files are removed from the controller and attached cluster storage."
            : "Completed and partial files are preserved. Resume continues from the existing data."
        }
      >
        <p>{control?.job.repo_id}</p>
        <Message error={action.error} />
        <div className="actions">
          <Button
            variant="outline"
            onClick={() => setControl(null)}
            disabled={action.busy}
          >
            Keep current state
          </Button>
          <Button
            variant={control?.action === "cancel" ? "destructive" : "default"}
            disabled={action.busy}
            onClick={() =>
              void action.run(async () => {
                await post(
                  `/api/v1/downloads/${control!.job.id}/${control!.action}`,
                );
                setControl(null);
              })
            }
          >
            Confirm {control?.action}
          </Button>
        </div>
      </Sheet>
      <Sheet
        open={hfOpen}
        onOpenChange={setHfOpen}
        title="Speed up downloads"
        description="Use a Hugging Face read token to increase request limits."
      >
        <a
          href="https://huggingface.co/settings/tokens"
          target="_blank"
          rel="noopener noreferrer"
        >
          Create a token ↗
        </a>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void action.run(async () => {
              await api("/api/v1/controller/huggingface", {
                method: "PUT",
                interactive: true,
                body: { token: token.trim() },
              });
              setToken("");
              setHfOpen(false);
            }, "Hugging Face token saved.");
          }}
        >
          <Field label="Hugging Face token">
            <Input
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              required
            />
          </Field>
          <Message error={action.error} />
          <Button disabled={action.busy}>Save token</Button>
        </form>
      </Sheet>
    </>
  );
}
