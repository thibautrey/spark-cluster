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
  Sheet,
  Empty,
} from "@/components/common";

import {
  ArrowUpRight,
  Cpu,
  Download,
  HardDrive,
  Heart,
  Search,
} from "lucide-react";

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

const workloadNames: Record<string, string> = {
  text: "Text & code",
  vision: "Vision & multimodal",
  embeddings: "Embeddings",
  classification: "Classification & ranking",
  speech: "Speech recognition",
  "text-to-image": "Image generation",
  "image-to-image": "Image editing",
  "text-to-video": "Video generation",
  "text-to-speech": "Speech synthesis",
};
const compactCount = (value?: number) =>
  value === undefined
    ? "—"
    : new Intl.NumberFormat(undefined, {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(value);
function RecipeArtwork({
  repository,
  workload,
  onOpen,
  title,
}: {
  repository: string;
  workload?: string;
  onOpen: () => void;
  title: string;
}) {
  const name = repository.split("/").at(-1) || repository;
  const initials = name
    .replace(/[^A-Za-z0-9]+/g, " ")
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
  return (
    <button
      className="recipe-artwork"
      type="button"
      onClick={onOpen}
      aria-label={`View ${title} recipe`}
    >
      <span
        className="catalog-cover recipe-cover"
        data-workload={workload}
        aria-hidden="true"
      >
        <span className="catalog-cover-fallback">{initials}</span>
        <img
          src={`https://cdn-thumbnails.huggingface.co/social-thumbnails/models/${repository.split("/").map(encodeURIComponent).join("/")}.png`}
          alt=""
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          onError={(event) => {
            event.currentTarget.hidden = true;
          }}
        />
        {workload && (
          <span className="recipe-workload">
            {workloadNames[workload] || workload}
          </span>
        )}
        <span className="recipe-artwork-open">
          <ArrowUpRight size={16} />
        </span>
      </span>
    </button>
  );
}
function RecipePublisher({ repository }: { repository: string }) {
  const owner = repository.split("/")[0];
  const [failed, setFailed] = useState(false);
  return (
    <div className="recipe-publisher">
      <span className="avatar" aria-hidden="true">
        {failed ? (
          owner.slice(0, 2).toUpperCase()
        ) : (
          <img
            src={`/api/v1/catalog/profiles/${encodeURIComponent(owner)}/avatar`}
            alt=""
            loading="lazy"
            referrerPolicy="no-referrer"
            onError={() => setFailed(true)}
          />
        )}
      </span>
      <span>{owner}</span>
    </div>
  );
}
function RecipePopularity({
  downloads,
  likes,
}: {
  downloads?: number;
  likes?: number;
}) {
  if (downloads === undefined && likes === undefined) return null;
  return (
    <div className="recipe-popularity">
      <span
        title={`${downloads?.toLocaleString() || 0} downloads in the popularity snapshot`}
      >
        <Download size={12} aria-hidden="true" /> {compactCount(downloads)}
      </span>
      <span title={`${likes?.toLocaleString() || 0} likes`}>
        <Heart size={12} aria-hidden="true" /> {compactCount(likes)}
      </span>
    </div>
  );
}
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
  const [detailId, setDetailId] = useState<string | null>(null);
  const [referenceId, setReferenceId] = useState<string | null>(null);
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
  const detailVersions = detailId ? groups.get(detailId) || [] : [];
  const detailEntry =
    detailVersions.find((e) => e.id === selected[detailId!]) ||
    detailVersions[0];
  const detailRecipe = detailEntry?.recipe;
  const reference = state.data?.references?.find(
    (entry) => entry.id === referenceId,
  );
  const snapshot = state.data?.catalog?.snapshots[0];
  const install = (entry: Entry) =>
    action.run(async () => {
      await post("/api/v1/recipes/install", {
        recipe_hash: entry.id,
        cluster_id: target!.id,
      });
      await refresh();
    });
  const installLabel = (recipe: Recipe) =>
    recipe.validation === "generated-template"
      ? "Install template & launch"
      : "Install & launch";
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
      <p className="note recipe-intro">
        Find a model setup for {target?.name || "your Spark"}. Choose a version,
        review its requirements, then install.
      </p>
      <div className="toolbar recipe-toolbar">
        <div className="recipe-search">
          <Search size={16} aria-hidden="true" />
          <Input
            aria-label="Search recipes"
            placeholder="Search model or recipe…"
            value={filters.q}
            onChange={(e) => updateFilters({ q: e.target.value })}
          />
        </div>
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
      </div>
      <div className="recipe-results-bar">
        <p className="note" aria-live="polite">
          {state.isFetching
            ? "Loading recipes…"
            : `${state.data?.total ?? groups.size} matching recipes`}
          {snapshot && (
            <span>
              {" "}
              · Updated {new Date(snapshot.captured_at).toLocaleDateString()}
            </span>
          )}
        </p>
        <label className="note">
          <input
            type="checkbox"
            checked={filters.installable}
            onChange={(e) => updateFilters({ installable: e.target.checked })}
          />{" "}
          Compatible with selected pool
        </label>
      </div>
      <Message error={state.error || action.error} />
      {state.data?.discovery.errors.map((error) => (
        <Message key={error}>{error}</Message>
      ))}
      <div className="card-grid recipe-grid">
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
            <article className="model-card recipe-card" key={id}>
              <RecipeArtwork
                repository={r.model.repository}
                workload={entry.origin.category}
                title={r.title}
                onOpen={() => setDetailId(id)}
              />
              <div className="recipe-card-body">
                <RecipePublisher repository={r.model.repository} />
                <h3>
                  {r.title === r.model.repository
                    ? r.model.repository.split("/").at(-1)
                    : r.title}
                </h3>
                <div className="recipe-status">
                  <Badge good={r.validation === "hardware-verified"}>
                    {r.validation === "hardware-verified"
                      ? "Publisher hardware verified"
                      : r.validation === "generated-template"
                        ? "Generated template · untested"
                        : "Source reviewed · untested on this pool"}
                  </Badge>
                  {installed && (
                    <Badge good>Installed v{installed.recipe.version}</Badge>
                  )}
                  {installed && installed.recipe_hash !== versions[0].id && (
                    <Badge>Update available</Badge>
                  )}
                </div>
                {r.validation !== "generated-template" && (
                  <p className="note recipe-description">{r.description}</p>
                )}
                <div className="recipe-specs">
                  <span title="Required Sparks">
                    <Cpu size={13} aria-hidden="true" /> {r.requirements.nodes}{" "}
                    Spark{r.requirements.nodes > 1 ? "s" : ""}
                  </span>
                  <span title="Memory budget per Spark">
                    <span className="recipe-spec-label">RAM</span>{" "}
                    {r.requirements.memory_gib_per_node} GiB
                  </span>
                  <span title="Storage budget per Spark">
                    <HardDrive size={13} aria-hidden="true" />{" "}
                    {r.requirements.disk_gib_per_node} GiB
                  </span>
                </div>
                <footer className="recipe-card-footer">
                  <RecipePopularity
                    downloads={entry.origin.downloads}
                    likes={entry.origin.likes}
                  />
                  <div className="recipe-release-row">
                    <Select
                      aria-label={`Version for ${r.title}`}
                      value={entry.id}
                      onChange={(e) =>
                        setSelected({ ...selected, [id]: e.target.value })
                      }
                    >
                      {versions.map((v) => (
                        <option key={v.id} value={v.id}>
                          v{v.recipe.version}
                        </option>
                      ))}
                    </Select>
                    <Button
                      variant="outline"
                      onClick={() => setDetailId(id)}
                      aria-label={`Details for ${r.title}`}
                    >
                      Details <ArrowUpRight size={13} aria-hidden="true" />
                    </Button>
                  </div>
                  {entry.blockers.length > 0 && (
                    <p
                      className="recipe-availability"
                      title={entry.blockers.join(" · ")}
                    >
                      {entry.blockers[0]}
                      {entry.blockers.length > 1
                        ? ` · +${entry.blockers.length - 1} more`
                        : ""}
                    </p>
                  )}
                  <Button
                    disabled={
                      action.busy || busy || !!entry.blockers.length || !target
                    }
                    onClick={() => install(entry)}
                  >
                    {running ? "Installing…" : installLabel(r)}
                  </Button>
                </footer>
              </div>
            </article>
          );
        })}
        {state.data?.references?.map((reference) => (
          <article className="model-card recipe-card" key={reference.id}>
            <RecipeArtwork
              repository={reference.model_id}
              workload={reference.task}
              title={reference.model_id}
              onOpen={() => setReferenceId(reference.id)}
            />
            <div className="recipe-card-body">
              <RecipePublisher repository={reference.model_id} />
              <h3>{reference.model_id.split("/").at(-1)}</h3>
              <div className="recipe-status">
                <Badge>Reference only · adapter required</Badge>
              </div>
              <p className="note recipe-description">{reference.blocker}</p>
              <footer className="recipe-card-footer">
                <RecipePopularity
                  downloads={reference.downloads}
                  likes={reference.likes}
                />
                <Button
                  variant="outline"
                  onClick={() => setReferenceId(reference.id)}
                >
                  View instructions{" "}
                  <ArrowUpRight size={13} aria-hidden="true" />
                </Button>
                <Button disabled>Runtime adapter required</Button>
              </footer>
            </div>
          </article>
        ))}
      </div>
      {!state.isFetching &&
        !groups.size &&
        !state.data?.references?.length &&
        !state.error && (
          <Empty>
            No recipes match these filters. Try another model or workload.
          </Empty>
        )}
      <div className="actions recipe-pagination">
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
          <p className="note">
            {state.data.catalog_total?.toLocaleString()} recipes ·{" "}
            {state.data.generated_total?.toLocaleString()} generated templates ·{" "}
            {state.data.catalog.reference_count.toLocaleString()} reference
            workloads. Generated templates are untested; architecture support
            does not prove Spark compatibility.
          </p>
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
      <Sheet
        open={!!detailRecipe}
        onOpenChange={(open) => {
          if (!open) setDetailId(null);
        }}
        title={detailRecipe?.title || "Recipe details"}
        description="Review the pinned release, requirements and compatibility before installing."
      >
        {detailRecipe && detailEntry && (
          <>
            <RecipePublisher repository={detailRecipe.model.repository} />
            <p>{detailRecipe.description}</p>
            <Select
              aria-label={`Review version for ${detailRecipe.title}`}
              value={detailEntry.id}
              onChange={(e) =>
                setSelected({ ...selected, [detailRecipe.id]: e.target.value })
              }
            >
              {detailVersions.map((v) => (
                <option key={v.id} value={v.id}>
                  v{v.recipe.version} · {v.id.slice(0, 8)}
                </option>
              ))}
            </Select>
            <div className="recipe-detail-facts">
              <div>
                <span>Sparks</span>
                <strong>{detailRecipe.requirements.nodes}</strong>
              </div>
              <div>
                <span>Memory / Spark</span>
                <strong>
                  {detailRecipe.requirements.memory_gib_per_node} GiB
                </strong>
              </div>
              <div>
                <span>Storage / Spark</span>
                <strong>
                  {detailRecipe.requirements.disk_gib_per_node} GiB
                </strong>
              </div>
            </div>
            <h3>Compatibility</h3>
            <p className="note">
              {detailRecipe.validation === "generated-template"
                ? "Generated template · untested on this pool."
                : detailRecipe.validation === "hardware-verified"
                  ? "Hardware verification was reported by the publisher."
                  : "Source reviewed · untested on this pool."}
            </p>
            {detailEntry.blockers.length ? (
              detailEntry.blockers.map((b) => <Message key={b}>{b}</Message>)
            ) : (
              <p className="good-text">Ready for installation preflight.</p>
            )}
            <h3>Model & runtime</h3>
            <p className="note">{detailRecipe.model.repository}</p>
            <p className="recipe-pin">
              Weights revision: {detailRecipe.model.revision}
            </p>
            <p className="recipe-pin">{detailRecipe.runtime.image}</p>
            <p>
              Recipe: {detailRecipe.license} · Model:{" "}
              {detailRecipe.model.license}
            </p>
            <div className="actions">
              {detailEntry.origin.model_card && (
                <a
                  href={detailEntry.origin.model_card}
                  target="_blank"
                  rel="noreferrer"
                >
                  Pinned model card ↗
                </a>
              )}
              <a
                href={`https://github.com/${detailRecipe.source.repository}/blob/${detailRecipe.source.commit}/${detailRecipe.source.path}`}
                target="_blank"
                rel="noreferrer"
              >
                Pinned upstream source ↗
              </a>
            </div>
            {detailRecipe.runtime.unconfined && (
              <Message>
                This recipe requests unconfined container seccomp/AppArmor.
              </Message>
            )}
            <ul className="recipe-detail-notes">
              {detailRecipe.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
            <p className="recipe-pin">Release hash: {detailEntry.id}</p>
            <Message error={action.error} />
            <Button
              disabled={
                action.busy || busy || !!detailEntry.blockers.length || !target
              }
              onClick={() => install(detailEntry)}
            >
              {installLabel(detailRecipe)}
            </Button>
          </>
        )}
      </Sheet>
      <Sheet
        open={!!reference}
        onOpenChange={(open) => {
          if (!open) setReferenceId(null);
        }}
        title={reference?.model_id || "Model instructions"}
        description="This workload requires a dedicated runtime adapter."
      >
        {reference && (
          <>
            <RecipePublisher repository={reference.model_id} />
            <Badge>Reference only · adapter required</Badge>
            <p>{reference.blocker}</p>
            <p>
              {reference.task} · {reference.library}
            </p>
            <p>Model license: {reference.license}</p>
            <p className="recipe-pin">Weights revision: {reference.revision}</p>
            <a href={reference.model_card} target="_blank" rel="noreferrer">
              Pinned model instructions ↗
            </a>
            <Button disabled>Runtime adapter required</Button>
          </>
        )}
      </Sheet>
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
