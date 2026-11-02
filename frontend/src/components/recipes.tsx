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
  origin: { kind: string; repository?: string; commit?: string };
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
  jobs: Job[];
  discovery: { checked_at: number | null; errors: string[]; stale?: boolean };
};

export function Recipes() {
  const { clusters } = useDashboardContext();
  const recoverStorage = useStorageRecovery();
  const target = clusters?.clusters.find(
    (c) => c.id === clusters.active_cluster_id,
  );
  const state = useApi<State>(
    clusters
      ? `/api/v1/recipes${target ? `?cluster_id=${encodeURIComponent(target.id)}` : ""}`
      : null,
    5000,
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
                {running ? "Installing…" : "Install & launch"}
              </Button>
            </article>
          );
        })}
      </div>
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
