import { useStorageRecovery, type SpaceRequest } from "@/components/storage";
import { useRef, useState } from "react";
import { post, useAction, useApi } from "@/lib/api";
import { useDashboardContext } from "@/lib/context";
import {
  Badge,
  Button,
  Message,
  Meter,
  Panel,
  Select,
  Input,
} from "@/components/common";

type Recipe = {
  id: string;
  version: string;
  title: string;
  description: string;
  validation: string;
  license: string;
  source: { repository: string; commit: string; path: string };
  model: { repository: string; revision: string; license: string };
  runtime: { image: string; unconfined: boolean };
  requirements: {
    nodes: number;
    memory_gib_per_node: number;
    disk_gib_per_node: number;
  };
  notes: string[];
};
type Entry = {
  id: string;
  recipe: Recipe;
  blockers: string[];
  origin: {
    kind: string;
    repository?: string;
    commit?: string;
    category?: string;
    task?: string;
    downloads?: number;
    likes?: number;
    trending_score?: number;
    captured_at?: string;
    model_card?: string;
  };
};
type Job = {
  id: string;
  cluster_id: string;
  recipe_hash: string;
  recipe: Recipe;
  status: string;
  stage: string;
  progress: number;
  error: string | null;
  storage?: SpaceRequest;
  artifacts_present?: boolean;
};
type State = {
  recipes: Entry[];
  references?: {
    id: string;
    model_id: string;
    revision: string;
    task: string;
    library: string;
    license: string;
    downloads: number;
    likes: number;
    model_card: string;
    blocker: string;
  }[];
  total?: number;
  catalog_total?: number;
  generated_total?: number;
  has_more?: boolean;
  catalog?: {
    reference_count: number;
    snapshots: { captured_at: string; ranking: string }[];
    upstream_libraries: { repository: string; commit: string; url: string }[];
  };
  jobs: Job[];
  discovery: { checked_at: number | null; errors: string[]; stale?: boolean };
};

export function Recipes() {
  const { clusters } = useDashboardContext();
  const recoverStorage = useStorageRecovery();
  const target = clusters?.clusters.find(
    (c) => c.id === clusters.active_cluster_id,
  );
  const [filters, setFilters] = useState({
    q: "",
    category: "all",
    sort: "popular",
    offset: 0,
    installable: false,
  });
  const updateFilters = (next: Partial<typeof filters>) =>
    setFilters((old) => ({ ...old, ...next, offset: 0 }));
  const params = new URLSearchParams({
    ...(target ? { cluster_id: target.id } : {}),
    q: filters.q,
    category: filters.category,
    sort: filters.sort,
    offset: String(filters.offset),
    limit: "24",
    installable: String(filters.installable),
  });
  const state = useApi<State>(
    clusters ? `/api/v1/recipes?${params}` : null,
    30000,
  );
  const action = useAction();
  const file = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<Record<string, string>>({});
  const groups = new Map<string, Entry[]>();
  for (const entry of state.data?.recipes || []) {
    groups.set(entry.recipe.id, [
      ...(groups.get(entry.recipe.id) || []),
      entry,
    ]);
  }
  const jobs = (state.data?.jobs || []).filter(
    (j) => j.cluster_id === target?.id,
  );
  const busy = jobs.some((j) => ["queued", "running"].includes(j.status));
  const refresh = () => state.refetch();
  return (
    <Panel
      title="Recipes"
      actions={
        <>
          <input
            hidden
            ref={file}
            type="file"
            accept=".json,application/json"
            aria-label="Import recipe file"
            onChange={(e) => {
              const upload = e.target.files?.[0];
              e.target.value = "";
              if (!upload) return;
              action.run(async () => {
                if (upload.size > 128 * 1024)
                  throw new Error("Recipe exceeds 128 KiB");
                await post(
                  "/api/v1/recipes/import",
                  JSON.parse(await upload.text()),
                );
                await refresh();
              });
            }}
          />
          <Button
            variant="outline"
            disabled={action.busy}
            onClick={() => file.current?.click()}
          >
            Import recipe
          </Button>
          <Button
            variant="outline"
            disabled={action.busy}
            onClick={() =>
              action.run(async () => {
                await post("/api/v1/recipes/discover");
                await refresh();
              })
            }
          >
            Discover on GitHub
          </Button>
        </>
      }
    >
      <p className="note">
        Versioned model and runtime setups for {target?.name || "your Spark"}.
        Choose a release, then install and launch. Downloads and compilation may
        take time.
      </p>
      {state.data?.catalog_total !== undefined && (
        <p className="note">
          {state.data.catalog_total.toLocaleString()} recipes ·{" "}
          {state.data.generated_total?.toLocaleString()} generated templates ·{" "}
          {state.data.catalog?.reference_count.toLocaleString()} reference
          workloads.
          {state.data.catalog?.snapshots[0] && (
            <>
              {" "}
              Popularity snapshot{" "}
              {new Date(
                state.data.catalog.snapshots[0].captured_at,
              ).toLocaleDateString()}
              .
            </>
          )}{" "}
          Generated templates are untested; architecture support does not prove
          Spark compatibility.
        </p>
      )}
      <div className="actions">
        <Input
          aria-label="Search recipes"
          placeholder="Search model or recipe…"
          value={filters.q}
          onChange={(e) => updateFilters({ q: e.target.value })}
        />
        <Select
          aria-label="Recipe workload"
          value={filters.category}
          onChange={(e) => updateFilters({ category: e.target.value })}
        >
          <option value="all">All serving recipes</option>
          <option value="curated">Reviewed recipes</option>
          <option value="text">Text, code & reasoning</option>
          <option value="vision">Vision & multimodal</option>
          <option value="embeddings">Embeddings & retrieval</option>
          <option value="classification">Classification & ranking</option>
          <option value="speech">Speech recognition</option>
          <option value="text-to-image">Image generation · references</option>
          <option value="image-to-image">Image editing · references</option>
          <option value="text-to-video">Video generation · references</option>
          <option value="text-to-speech">Speech synthesis · references</option>
        </Select>
        <Select
          aria-label="Sort recipes"
          value={filters.sort}
          onChange={(e) => updateFilters({ sort: e.target.value })}
        >
          <option value="popular">Popularity across tasks</option>
          <option value="trending">Trending now</option>
          <option value="downloads">Most downloaded</option>
          <option value="name">Name</option>
        </Select>
        <label className="note">
          <input
            type="checkbox"
            checked={filters.installable}
            onChange={(e) => updateFilters({ installable: e.target.checked })}
          />{" "}
          Compatible with selected pool
        </label>
      </div>
      <p className="note" aria-live="polite">
        {state.isFetching
          ? "Loading recipes…"
          : `${state.data?.total ?? groups.size} matching recipes`}
      </p>
      <Message error={state.error || action.error} />
      {state.data?.discovery.errors.map((error) => (
        <Message key={error}>{error}</Message>
      ))}
      <div className="model-grid">
        {[...groups.entries()].map(([id, versions]) => {
          const entry =
            versions.find((v) => v.id === selected[id]) || versions[0];
          const r = entry.recipe;
          const installed = jobs.find(
            (j) =>
              j.recipe.id === id &&
              j.status === "completed" &&
              j.artifacts_present !== false,
          );
          const running = jobs.find(
            (j) =>
              j.recipe_hash === entry.id &&
              ["queued", "running"].includes(j.status),
          );
          return (
            <article className="model-card" key={id}>
              <h3>{r.title}</h3>
              <p>{r.description}</p>
              <div className="actions">
                <Badge>
                  {r.validation === "hardware-verified"
                    ? "Publisher hardware verified"
                    : r.validation === "generated-template"
                      ? "Generated template · untested"
                      : "Source reviewed · untested on this pool"}
                </Badge>
                <Badge>{entry.origin.kind}</Badge>
                {installed && (
                  <Badge good>Installed v{installed.recipe.version}</Badge>
                )}
                {installed && installed.recipe_hash !== versions[0].id && (
                  <Badge>Another release available</Badge>
                )}
              </div>
              {entry.origin.task && (
                <p className="note">
                  {entry.origin.task} ·{" "}
                  {entry.origin.downloads?.toLocaleString()} downloads / 30 days
                  · {entry.origin.likes?.toLocaleString()} likes
                </p>
              )}
              <p className="note">
                {r.requirements.nodes} Spark ·{" "}
                {r.requirements.memory_gib_per_node} GiB memory ·{" "}
                {r.requirements.disk_gib_per_node} GiB storage budget
              </p>
              <Select
                aria-label={`Version for ${r.title}`}
                value={entry.id}
                onChange={(e) =>
                  setSelected({ ...selected, [id]: e.target.value })
                }
              >
                {versions.map((v) => (
                  <option key={v.id} value={v.id}>
                    v{v.recipe.version} · {v.id.slice(0, 8)}
                  </option>
                ))}
              </Select>
              <details>
                <summary>Recipe details and licenses</summary>
                <p>
                  {r.model.repository} · revision {r.model.revision}
                </p>
                {entry.origin.model_card && (
                  <a
                    href={entry.origin.model_card}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Pinned model card ↗
                  </a>
                )}
                <p className="note">{r.runtime.image}</p>
                <p>
                  Recipe: {r.license} · Model: {r.model.license}
                </p>
                <a
                  href={`https://github.com/${r.source.repository}/blob/${r.source.commit}/${r.source.path}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Pinned upstream source ↗
                </a>
                {r.runtime.unconfined && (
                  <p className="note">
                    This recipe requests unconfined container seccomp/AppArmor.
                  </p>
                )}
                <ul>
                  {r.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
                <p className="note">Release hash: {entry.id}</p>
              </details>
              {entry.blockers.map((b) => (
                <Message key={b}>{b}</Message>
              ))}
              <Button
                disabled={
                  action.busy || busy || !!entry.blockers.length || !target
                }
                onClick={() =>
                  action.run(async () => {
                    await post("/api/v1/recipes/install", {
                      recipe_hash: entry.id,
                      cluster_id: target!.id,
                    });
                    await refresh();
                  })
                }
              >
                {running
                  ? "Installing…"
                  : r.validation === "generated-template"
                    ? "Install template & launch"
                    : "Install & launch"}
              </Button>
            </article>
          );
        })}
        {state.data?.references?.map((reference) => (
          <article className="model-card" key={reference.id}>
            <h3>{reference.model_id}</h3>
            <Badge>Reference only · adapter required</Badge>
            <p>
              {reference.task} · {reference.library}
            </p>
            <p className="note">
              {reference.downloads.toLocaleString()} downloads / 30 days ·{" "}
              {reference.likes.toLocaleString()} likes
            </p>
            <p>{reference.blocker}</p>
            <details>
              <summary>Pinned source and license</summary>
              <p>Weights revision: {reference.revision}</p>
              <p>Model license: {reference.license}</p>
              <a href={reference.model_card} target="_blank" rel="noreferrer">
                Pinned model instructions ↗
              </a>
            </details>
            <Button disabled>Runtime adapter required</Button>
          </article>
        ))}
      </div>
      <div className="actions">
        <Button
          variant="outline"
          disabled={filters.offset === 0 || state.isFetching}
          onClick={() =>
            setFilters((old) => ({
              ...old,
              offset: Math.max(0, old.offset - 24),
            }))
          }
        >
          Previous recipes
        </Button>
        <span className="note">Page {Math.floor(filters.offset / 24) + 1}</span>
        <Button
          variant="outline"
          disabled={!state.data?.has_more || state.isFetching}
          onClick={() =>
            setFilters((old) => ({ ...old, offset: old.offset + 24 }))
          }
        >
          Next recipes
        </Button>
      </div>
      {state.data?.catalog && (
        <details>
          <summary>Catalog sources and ranking</summary>
          <p className="note">{state.data.catalog.snapshots[0]?.ranking}</p>
          <p className="note">
            Templates are generated from pinned model metadata and the runtime
            architecture registry. References are not counted as executable
            serving templates.
          </p>
          {state.data.catalog.upstream_libraries.map((source) => (
            <p key={source.repository}>
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.repository} ↗
              </a>{" "}
              · {source.commit.slice(0, 12)}
            </p>
          ))}
        </details>
      )}
      {jobs.slice(0, 5).map((job) => (
        <div key={job.id} className="download-card">
          <strong>
            {job.recipe.title} · v{job.recipe.version}
          </strong>
          <p>
            {job.status} · {job.stage}
          </p>
          <Meter value={job.progress} label="Recipe installation" />
          <Message error={job.error} />
          {job.status === "failed" &&
            (job.storage || /space|disk|storage/i.test(job.error || "")) && (
              <Button
                variant="outline"
                disabled={action.busy || busy}
                onClick={() =>
                  action.run(async () => {
                    if (
                      await recoverStorage({ space: job.storage, retry: true })
                    ) {
                      await post("/api/v1/recipes/install", {
                        recipe_hash: job.recipe_hash,
                        cluster_id: job.cluster_id,
                      });
                      await refresh();
                    }
                  })
                }
              >
                Free disk space and retry
              </Button>
            )}

          {job.status === "failed" && (
            <Button
              variant="outline"
              disabled={action.busy || busy}
              onClick={() =>
                action.run(async () => {
                  await post("/api/v1/recipes/install", {
                    recipe_hash: job.recipe_hash,
                    cluster_id: job.cluster_id,
                  });
                  await refresh();
                })
              }
            >
              Retry this release
            </Button>
          )}
        </div>
      ))}
      <p className="note">
        {state.data?.discovery.checked_at
          ? `GitHub checked ${new Date(state.data.discovery.checked_at * 1000).toLocaleString()}. Discoveries are cached for one hour.`
          : "GitHub discovery runs in the background. Bundled recipes are available offline."}{" "}
        Existing releases stay available for rollback. Unload running models
        before changing recipes.
      </p>
    </Panel>
  );
}
