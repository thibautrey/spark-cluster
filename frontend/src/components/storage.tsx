import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { api, installStoragePrompt, post, useAction, useApi } from "@/lib/api";
import {
  Badge,
  Button,
  Message,
  Panel,
  Select,
  Sheet,
} from "@/components/common";
import { bytes } from "@/lib/utils";

export type SpaceRequest = {
  requirements: {
    location_id: string;
    name: string;
    required_bytes: number;
    available_bytes: number;
    missing_bytes: number;
  }[];
  excluded_model_ids: string[];
};
type Location = {
  id: string;
  cluster_id: string;
  cluster_name: string;
  name: string;
  paths: string[];
  free_bytes: number | null;
  total_bytes: number | null;
  scope: string;
  node_ids: string[];
  node_names: string[];
  kind: string;
  scanning?: boolean;
};
type Asset = {
  id: string;
  location_id: string;
  name: string;
  model_id: string | null;
  revision?: string;
  path: string;
  kind: string;
  size_bytes: number;
  reclaimable_bytes: number;
  blockers: string[];
  recipe_refs: { id: string; version: string }[];
};
type Inventory = { locations: Location[]; assets: Asset[]; errors: string[] };
type OpenRequest = {
  space?: SpaceRequest;
  selected?: string[];
  retry?: boolean;
};
const StorageContext = createContext<
  (request?: OpenRequest) => Promise<boolean>
>(async () => false);
export const useStorageRecovery = () => useContext(StorageContext);

export function StorageRecoveryProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<OpenRequest | null>(null);
  const [resolve, setResolve] = useState<((value: boolean) => void) | null>(
    null,
  );
  function open(value: OpenRequest = {}) {
    return new Promise<boolean>((done) => {
      setRequest(value);
      setResolve(() => done);
    });
  }
  useEffect(
    () => installStoragePrompt((space) => open({ space, retry: true })),
    [],
  );
  function close(retry: boolean) {
    resolve?.(retry);
    setRequest(null);
    setResolve(null);
  }
  return (
    <StorageContext.Provider value={open}>
      {children}
      {request && <StorageChooser request={request} close={close} />}
    </StorageContext.Provider>
  );
}

function assetDescription(asset: Asset) {
  return `${bytes(asset.size_bytes)} on disk · ${bytes(asset.reclaimable_bytes)} potentially reclaimable`;
}

export function StoragePanel() {
  const storage = useApi<Inventory>("/api/v1/storage", 15000);
  const open = useStorageRecovery();
  const action = useAction();
  const [node, setNode] = useState("all");
  const nodes = new Map<string, string>();
  for (const location of storage.data?.locations || []) {
    location.node_ids.forEach((id, i) =>
      nodes.set(
        `${location.cluster_id}|${id}`,
        `${location.cluster_name} · ${location.node_names[i] || id}`,
      ),
    );
  }
  const locations = (storage.data?.locations || []).filter(
    (l) =>
      node === "all" ||
      l.node_ids.some((id) => `${l.cluster_id}|${id}` === node),
  );
  return (
    <Panel
      title="Stored on disk"
      actions={
        <Button
          variant="outline"
          disabled={action.busy}
          onClick={() =>
            action.run(() => api("/api/v1/storage", { interactive: true }))
          }
        >
          Refresh storage
        </Button>
      }
    >
      <p className="note">
        Weights stored on each node remain here when a model is unloaded. Shared
        storage is counted once.
      </p>
      <Select
        aria-label="Node storage"
        value={node}
        onChange={(e) => setNode(e.target.value)}
      >
        <option value="all">All nodes and controller</option>
        {[...nodes].map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </Select>
      <Message error={storage.error || action.error} />
      {storage.data?.errors.map((error) => (
        <Message key={error}>{error}</Message>
      ))}
      {locations.map((location) => (
        <section key={location.id} className="model-card">
          <h3>
            {location.cluster_name} · {location.name}
          </h3>
          {location.scanning && (
            <p role="status" className="note">
              Scanning stored files… sizes will appear when the disk scan
              completes.
            </p>
          )}
          <p>
            {location.free_bytes === null
              ? location.kind === "docker"
                ? "Docker free capacity is not reported"
                : "Disk capacity is being checked"
              : `${bytes(location.free_bytes)} free of ${bytes(location.total_bytes || 0)}`}
            {location.scope === "shared" && " · Shared across nodes"}
          </p>
          <small>{location.node_names.join(", ")}</small>
          {storage.data?.assets
            .filter((a) => a.location_id === location.id)
            .map((asset) => (
              <article className="list-row" key={asset.id}>
                <div>
                  <strong>{asset.name}</strong> <Badge>{asset.kind}</Badge>
                  <p className="note">{assetDescription(asset)}</p>
                  <small>{asset.path}</small>
                  {!!asset.recipe_refs.length && (
                    <p className="note">
                      Used by{" "}
                      {asset.recipe_refs
                        .map((r) => `${r.id} v${r.version}`)
                        .join(", ")}
                    </p>
                  )}
                  {asset.blockers.map((b) => (
                    <p className="note" key={b}>
                      {b}
                    </p>
                  ))}
                </div>
                <Button
                  variant="outline"
                  disabled={!!asset.blockers.length}
                  onClick={() => open({ selected: [asset.id] })}
                >
                  Delete files…
                </Button>
              </article>
            ))}
          {!storage.data?.assets.some((a) => a.location_id === location.id) && (
            <p className="note">
              No model weights or recipe artifacts on this storage.
            </p>
          )}
        </section>
      ))}
    </Panel>
  );
}

function StorageChooser({
  request,
  close,
}: {
  request: OpenRequest;
  close: (retry: boolean) => void;
}) {
  const state = useApi<Inventory>("/api/v1/storage");
  const client = useQueryClient();
  const action = useAction();
  const [selected, setSelected] = useState(new Set(request.selected || []));
  const [deleted, setDeleted] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const required = request.space?.requirements || [];
  const requiredLocations = new Set(required.map((r) => r.location_id));
  const assets = state.data?.assets || [];
  const selectedAssets = assets.filter((a) => selected.has(a.id));
  const canRetry =
    !required.length ||
    required.every((r) => {
      const location = state.data?.locations.find(
        (l) => l.id === r.location_id,
      );
      return (
        location?.free_bytes != null && location.free_bytes >= r.required_bytes
      );
    });
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open && !action.busy) close(false);
      }}
      title={
        request.retry ? "Free disk space to continue" : "Delete stored files"
      }
      description="Select the exact copies to remove. Deletion is permanent; other node copies and recipe definitions remain available."
    >
      <Message error={state.error || action.error} />
      {errors.map((error) => (
        <Message key={error} error={error} />
      ))}
      {required.map((r) => {
        const location = state.data?.locations.find(
          (l) => l.id === r.location_id,
        );
        const free = location?.free_bytes ?? r.available_bytes;
        const missing = Math.max(0, r.required_bytes - free);
        const reclaim = selectedAssets
          .filter((a) => a.location_id === r.location_id)
          .reduce((sum, a) => sum + a.reclaimable_bytes, 0);
        return (
          <div key={r.location_id} className="model-card">
            <strong>{r.name}</strong>
            <p>
              {bytes(r.required_bytes)} needed · {bytes(free)} free ·{" "}
              {bytes(missing)} short
            </p>
            <p className="note">
              Selected here: up to {bytes(reclaim)}. Free space is measured
              again after deletion.
            </p>
          </div>
        );
      })}
      {state.data?.errors.map((error) => (
        <Message key={error}>{error}</Message>
      ))}
      {state.data?.locations.map((location) => (
        <section key={location.id}>
          <h3>
            {location.cluster_name} · {location.name}
            {requiredLocations.has(location.id) ? " · Space needed here" : ""}
          </h3>
          <p className="note">
            {location.free_bytes == null
              ? "Capacity unknown; image layers may be shared"
              : `${bytes(location.free_bytes)} free`}
            {location.scope === "shared" ? " · Shared across nodes" : ""}
          </p>
          {assets
            .filter((a) => a.location_id === location.id)
            .map((asset) => {
              const excluded =
                !!asset.model_id &&
                request.space?.excluded_model_ids.includes(asset.model_id);
              const blocked = !!asset.blockers.length || !!excluded;
              return (
                <label key={asset.id} className="list-row">
                  <input
                    type="checkbox"
                    aria-label={`Delete ${asset.name} from ${location.cluster_name}`}
                    checked={selected.has(asset.id)}
                    disabled={blocked || action.busy}
                    onChange={(e) =>
                      setSelected((previous) => {
                        const next = new Set(previous);
                        if (e.target.checked) next.add(asset.id);
                        else next.delete(asset.id);
                        return next;
                      })
                    }
                  />
                  <div>
                    <strong>{asset.name}</strong>
                    <p>{assetDescription(asset)}</p>
                    <small>{asset.path}</small>
                    {!!asset.recipe_refs.length && (
                      <p className="note">
                        Shared by recipes:{" "}
                        {asset.recipe_refs
                          .map((r) => `${r.id} v${r.version}`)
                          .join(", ")}
                      </p>
                    )}
                    {asset.blockers.map((b) => (
                      <p className="note" key={b}>
                        {b}
                      </p>
                    ))}
                    {excluded && (
                      <p className="note">
                        Needed by the operation you are trying to run
                      </p>
                    )}
                  </div>
                </label>
              );
            })}
        </section>
      ))}
      {deleted && (
        <Message>Selected files removed. Storage has been refreshed.</Message>
      )}
      <p className="note">
        Deleting weights affects every recipe that shares them. Recipe files and
        version history are kept. Runtime images and caches may live on a
        different disk.
      </p>
      <div className="actions">
        <Button
          variant="outline"
          disabled={action.busy}
          onClick={() => close(false)}
        >
          Close
        </Button>
        <Button
          variant="destructive"
          disabled={
            action.busy ||
            !selectedAssets.length ||
            selectedAssets.some((a) => a.blockers.length)
          }
          onClick={() =>
            action.run(async () => {
              const result = await post<{
                deleted: string[];
                errors: string[];
              }>("/api/v1/storage/delete", {
                asset_ids: selectedAssets.map((a) => a.id),
              });
              setErrors(result.errors);
              setSelected(new Set());
              setDeleted(result.deleted.length > 0);
              await client.invalidateQueries();
              await state.refetch();
            })
          }
        >
          Delete selected files permanently
        </Button>
        {request.retry && (
          <Button
            disabled={action.busy || !canRetry}
            onClick={() => close(true)}
          >
            Retry operation
          </Button>
        )}
      </div>
    </Sheet>
  );
}
