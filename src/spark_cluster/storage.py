"""Physical model storage, scoped deletion, and recoverable disk admission errors."""

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import stat
import threading
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from .model_manager import WEIGHT_SUFFIXES
from .remote_cluster import signed_controller_headers

GIB = 2**30
ACTIVE_DOWNLOADS = {
    "queued",
    "resolving",
    "downloading",
    "paused",
    "preparing-transfer",
    "distributing",
}


class StorageShortage(ValueError):
    def __init__(self, requirements, excluded_models=()):
        self.storage = {"requirements": requirements, "excluded_model_ids": list(excluded_models)}
        super().__init__(
            "Not enough disk space. Choose stored models or recipe artifacts to remove."
        )

    def detail(self):
        return {"message": str(self), "code": "insufficient_storage", "storage": self.storage}


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()


def measure(path):
    """Count allocated blocks once; external hard links/symlink targets aren't reclaimable."""
    files = (
        [path]
        if path.is_file()
        else (
            Path(root) / name
            for root, _, names in os.walk(path, followlinks=False)
            for name in names
        )
    )
    inodes = {}
    logical = 0
    for file in files:
        info = file.lstat()
        if stat.S_ISLNK(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        key = (info.st_dev, info.st_ino)
        previous = inodes.get(key)
        if previous:
            previous[1] += 1
        else:
            inodes[key] = [info, 1]
            logical += info.st_size
    reclaimable = sum(
        info.st_blocks * 512 for info, count in inodes.values() if count >= info.st_nlink
    )
    signature = token(
        str(
            sorted(
                (key, info.st_size, info.st_mtime_ns, info.st_nlink)
                for key, (info, _) in inodes.items()
            )
        )
    )
    allocated = sum(info.st_blocks * 512 for info, _ in inodes.values())
    return allocated, reclaimable, signature


def weight_directory(path):
    return path.is_dir() and (
        any(path.glob("*.safetensors")) or any(path.glob("*.gguf")) or any(path.glob("*.bin"))
    )


class StorageService:
    def __init__(self, settings, store, registry, node, library):
        self.settings, self.store, self.registry = settings, store, registry
        self.node, self.library = node, library
        self.transport = None
        self.lock = node.storage_lock
        node.models.storage_lock = self.lock
        node.disk_admission = self.local_launch_check
        self.cached = None
        self.cached_at = 0
        self.on_change = None
        self.scan_future = None
        self.scan_state = None
        self.scan_at = 0

    def _scan(self):
        root = self.settings.model_cache_path
        root.mkdir(parents=True, exist_ok=True)
        roots = [("models", root)]
        hf = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        if hf.is_dir() and hf.resolve() != root.resolve():
            roots.append(("huggingface", hf))
        locations, assets, paths = {}, [], {}
        model_location = None
        for kind, base in roots:
            base = base.resolve()
            device = str(base.stat().st_dev)
            location_id = "disk-" + token(device)[:16]
            usage = shutil.disk_usage(base)
            location = locations.setdefault(
                location_id,
                {
                    "id": location_id,
                    "name": "Model disk",
                    "paths": [],
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                    "kind": "filesystem",
                    "scope": "node",
                },
            )
            location["paths"].append(str(base))
            if kind == "models":
                model_location = location_id
                candidates = []
                for first in base.iterdir():
                    if first.name.startswith(".") or first.is_symlink():
                        continue
                    if first.is_file() and first.suffix.lower() in WEIGHT_SUFFIXES:
                        candidates.append((first, first.stem, None, "weights"))
                    elif first.is_dir():
                        dirs = (
                            [first]
                            if (first / ".spark-cluster.json").is_file() or weight_directory(first)
                            else list(first.iterdir())
                        )
                        for path in dirs:
                            if not path.is_dir() or path.is_symlink():
                                continue
                            manifest = path / ".spark-cluster.json"
                            if manifest.is_file():
                                try:
                                    data = json.loads(manifest.read_text())
                                    candidates.append(
                                        (path, data["repo_id"], data.get("revision"), "weights")
                                    )
                                except (ValueError, KeyError, OSError):
                                    continue
                            elif weight_directory(path):
                                candidates.append(
                                    (path, path.relative_to(base).as_posix(), None, "weights")
                                )
            else:
                candidates = [
                    (
                        p,
                        p.name.removeprefix("models--").replace("--", "/", 1),
                        None,
                        "huggingface-cache",
                    )
                    for p in base.glob("models--*")
                    if p.is_dir() and not p.is_symlink()
                ]
            for path, model_id, revision, asset_kind in candidates:
                if not path.resolve().is_relative_to(base) or path.resolve() == base:
                    continue
                logical, reclaimable, signature = measure(path)
                info = path.stat()
                identity = token(f"{path}:{info.st_ino}:{info.st_mtime_ns}:{revision}:{signature}")
                assets.append(
                    {
                        "id": identity,
                        "location_id": location_id,
                        "kind": asset_kind,
                        "name": model_id,
                        "model_id": model_id,
                        "revision": revision,
                        "path": str(path),
                        "size_bytes": logical,
                        "reclaimable_bytes": reclaimable,
                        "recipe_refs": [],
                        "blockers": [],
                    }
                )
                paths[identity] = (path, base)
        return {
            "locations": list(locations.values()),
            "assets": assets,
            "model_location_id": model_location,
        }, paths

    async def _busy(self):
        models = {
            r.profile.model_id or r.profile.model for r in self.node.demand.residents.values()
        }
        if not self.settings.remote_cluster_url and not self.node.local_cluster():
            endpoint = await self.node.models.runtime_endpoint()
            models.update(endpoint.loaded_model_ids)
        if self.node.native.get("model_id"):
            models.add(self.node.native["model_id"])
        for job in await self.node.deployments.launches():
            if job.status not in {"failed", "ready"}:
                models.add(job.request.model_id)
            elif job.status == "ready" and not self.node.local_cluster():
                # Kubernetes/native runtimes may be serving without a Docker residency entry.
                endpoint = await self.node.models.runtime_endpoint()
                models.update(endpoint.loaded_model_ids)
        downloads = await asyncio.to_thread(self.store.list_downloads)
        models.update(j.repo_id for j in downloads if j.status in ACTIVE_DOWNLOADS)
        installs = await asyncio.to_thread(self.store.runtime_objects, "recipe-install")
        for job in installs:
            if job["status"] in {"queued", "running"}:
                models.add(job["recipe"]["model"]["repository"])
        return models, installs

    def _quick_disk_state(self):
        root = self.settings.model_cache_path.resolve()
        root.mkdir(parents=True, exist_ok=True)
        identity = "disk-" + token(str(root.stat().st_dev))[:16]
        usage = shutil.disk_usage(root)
        return {
            "locations": [
                {
                    "id": identity,
                    "name": "Model disk",
                    "paths": [str(root)],
                    "free_bytes": usage.free,
                    "total_bytes": usage.total,
                    "kind": "filesystem",
                    "scope": "node",
                }
            ],
            "assets": [],
            "model_location_id": identity,
        }

    async def _file_inventory(self):
        # A daemon thread cannot hold controller shutdown hostage to an unavailable NAS.
        if self.scan_future is None or (
            self.scan_future.done() and time.monotonic() - self.scan_at > 30
        ):
            future = concurrent.futures.Future()
            self.scan_future = future

            def scan():
                try:
                    result = self._scan()
                    self.scan_at = time.monotonic()
                    future.set_result(result)
                except Exception as error:
                    future.set_exception(error)

            threading.Thread(target=scan, daemon=True, name="storage-inventory").start()
        try:
            state, _ = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self.scan_future)), timeout=0.25
            )
            self.scan_state = state
            scanning = False
        except TimeoutError:
            state = self.scan_state or await asyncio.to_thread(self._quick_disk_state)
            scanning = True
        result = copy.deepcopy(state)
        for location in result["locations"]:
            usage = await asyncio.to_thread(shutil.disk_usage, location["paths"][0])
            location.update(free_bytes=usage.free, total_bytes=usage.total, scanning=scanning)
        result["scanning"] = scanning
        return result

    def _resolve_selection(self, selected):
        paths = {}
        for asset in selected:
            if asset["kind"] not in {"weights", "huggingface-cache"}:
                continue
            path = Path(asset["path"])
            root = (
                self.settings.model_cache_path
                if asset["kind"] == "weights"
                else Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
            ).resolve()
            if (
                path.is_symlink()
                or path.resolve() == root
                or not path.resolve().is_relative_to(root)
            ):
                raise ValueError("Unsafe storage path")
            _, _, signature = measure(path)
            revision = asset.get("revision")
            if (path / ".spark-cluster.json").is_file():
                revision = json.loads((path / ".spark-cluster.json").read_text()).get("revision")
            info = path.stat()
            identity = token(f"{path}:{info.st_ino}:{info.st_mtime_ns}:{revision}:{signature}")
            if identity != asset["id"]:
                raise ValueError("Stored files changed; refresh before deleting")
            paths[identity] = (path, root)
        return paths

    async def local(self, *, fresh=False):
        if not fresh and self.cached and time.monotonic() - self.cached_at < 10:
            return self.cached
        value = await self._file_inventory()
        busy, installs = await self._busy()
        local = self.node.local_cluster()
        nodes = (
            [] if self.settings.remote_cluster_url and not local else self.node.machine_inventory()
        )
        node_ids = [m["id"] for m in nodes] or [local.host_address if local else "controller"]
        names = [m["name"] for m in nodes] or [local.name if local else "Controller"]
        for location in value["locations"]:
            location.update(
                node_ids=node_ids,
                node_names=names,
                scope="shared" if not local and len(node_ids) > 1 else "node",
            )
        for asset in value["assets"]:
            if asset["model_id"] in busy:
                asset["blockers"].append("In use by a runtime, download, or recipe installation")
            asset["recipe_refs"] = [
                {"id": j["recipe"]["id"], "version": j["recipe"]["version"]}
                for j in installs
                if j["recipe"]["model"]["repository"] == asset["model_id"]
            ]
        value["errors"] = []
        if self.library.docker.available:
            try:
                await self._docker_assets(value, installs)
            except (httpx.HTTPError, RuntimeError, ValueError):
                value["errors"].append("Docker artifact usage unavailable; deletion is disabled")
                for asset in value["assets"]:
                    asset["blockers"].append("Cannot verify Docker runtime usage")
        self.cached, self.cached_at = value, time.monotonic()
        return value

    async def _docker_assets(self, value, installs):
        docker = self.library.docker
        # Recipe runtime images and cache volumes are catalogued, never unrelated Docker data.
        refs = {}
        for job in installs:
            recipe = job["recipe"]
            refs.setdefault(recipe["runtime"]["image"], []).append(
                {"id": recipe["id"], "version": recipe["version"]}
            )
        containers = (
            await docker.request("GET", "/containers/json", params={"all": "true"})
        ).json()
        protected_models = set()
        for container in containers:
            if container.get("State") == "running":
                model = container.get("Labels", {}).get("spark-cluster.model-id")
                if model:
                    protected_models.add(model)
        for asset in value["assets"]:
            if asset["model_id"] in protected_models:
                asset["blockers"].append("Model is mounted by a running container")
        # Legacy/manual GPU runtimes may not have a recipe installation record.
        for container in containers:
            if container.get("State") != "running" or container.get("Labels", {}).get(
                "spark-cluster.model-id"
            ):
                continue
            details = (await docker.request("GET", f"/containers/{container['Id']}/json")).json()
            host_config = details.get("HostConfig", {})
            if not host_config.get("DeviceRequests") and host_config.get("Runtime") != "nvidia":
                continue
            for asset in value["assets"]:
                path = Path(asset["path"])
                host_path = (
                    self.settings.docker_host_model_path
                    / path.relative_to(self.settings.model_cache_path.resolve())
                    if asset["kind"] == "weights"
                    else path
                )
                for mount in details.get("Mounts", []):
                    source = Path(mount.get("Source") or "/nonexistent-spark-source")
                    if host_path.is_relative_to(source) or source.is_relative_to(host_path):
                        asset["blockers"].append("Files may be used by an external GPU container")
        if not refs:
            return
        images = (await docker.request("GET", "/system/df", timeout=10)).json()
        value["locations"].append(
            {
                "id": "docker",
                "name": "Docker images and recipe caches",
                "kind": "docker",
                "paths": [],
                "scope": "node",
                "node_ids": value["locations"][0]["node_ids"],
                "node_names": value["locations"][0]["node_names"],
                "free_bytes": None,
                "total_bytes": None,
            }
        )
        for image in images.get("Images") or []:
            matching = [
                ref
                for ref in refs
                if ref.removeprefix("docker.io/")
                in {r.removeprefix("docker.io/") for r in image.get("RepoDigests") or []}
            ]
            if not matching:
                continue
            used = any(c.get("ImageID") == image["Id"] for c in containers)
            active_install = any(
                j["status"] in {"queued", "running"} and j["recipe"]["runtime"]["image"] in matching
                for j in installs
            )
            value["assets"].append(
                {
                    "id": "image-" + token(image["Id"]),
                    "location_id": "docker",
                    "kind": "runtime-image",
                    "name": matching[0],
                    "model_id": None,
                    "path": image["Id"],
                    "size_bytes": image.get("Size", 0),
                    "reclaimable_bytes": max(
                        0, image.get("Size", 0) - max(0, image.get("SharedSize", 0))
                    ),
                    "recipe_refs": [r for ref in matching for r in refs[ref]],
                    "blockers": ["Referenced by a container or active installation"]
                    if used or active_install
                    else [],
                }
            )
        volumes = {"spark-recipe-" + token(image)[:16]: recipes for image, recipes in refs.items()}
        for volume in images.get("Volumes") or []:
            if volume["Name"] not in volumes:
                continue
            usage = volume.get("UsageData") or {}
            count = usage.get("RefCount", -1)
            value["assets"].append(
                {
                    "id": "volume-" + token(volume["Name"]),
                    "location_id": "docker",
                    "kind": "recipe-cache",
                    "name": volume["Name"],
                    "model_id": None,
                    "path": volume["Name"],
                    "size_bytes": max(0, usage.get("Size", 0)),
                    "reclaimable_bytes": max(0, usage.get("Size", 0)),
                    "recipe_refs": volumes[volume["Name"]],
                    "blockers": ["Cache is attached to a container or its usage is unknown"]
                    if count != 0
                    else [],
                }
            )

    async def _remote(self, cluster, method, suffix, payload=None):
        if not self.settings.model_admin_token:
            raise ValueError("Configure the shared administration token for remote storage")
        path = "/api/v1/storage/node" + suffix
        body = json.dumps(payload).encode() if payload is not None else b""
        headers = signed_controller_headers(self.settings.model_admin_token, method, path, body)
        headers["content-type"] = "application/json"
        async with httpx.AsyncClient(
            base_url=cluster.endpoint,
            transport=self.transport,
            timeout=20 if method == "GET" else 120,
            trust_env=False,
        ) as client:
            response = await client.request(method, path, headers=headers, content=body)
            response.raise_for_status()
            return response.json()

    async def inventory(self, *, fresh=False):
        local = await self.local(fresh=fresh)
        target = self.node.local_cluster()
        result = {"locations": [], "assets": [], "errors": list(local["errors"])}
        self._merge(
            result,
            local,
            target.id if target else "controller",
            target.name if target else "Controller",
        )

        async def remote(cluster):
            try:
                state = await self._remote(cluster, "GET", "")
                self._merge(result, state, cluster.id, cluster.name)
            except (httpx.HTTPError, ValueError, KeyError):
                result["errors"].append(
                    f"{cluster.name}: storage unavailable; update or reconnect its bridge"
                )

        await asyncio.gather(
            *(remote(c) for c in self.registry.list() if c.kind == "remote-bridge")
        )
        return result

    @staticmethod
    def _merge(result, state, cluster_id, name):
        for location in state["locations"]:
            result["locations"].append(
                {
                    **location,
                    "id": cluster_id + ":" + location["id"],
                    "cluster_id": cluster_id,
                    "cluster_name": name,
                }
            )
        for asset in state["assets"]:
            result["assets"].append(
                {
                    **asset,
                    "id": cluster_id + ":" + asset["id"],
                    "location_id": cluster_id + ":" + asset["location_id"],
                    "cluster_id": cluster_id,
                }
            )
        result["errors"].extend(f"{name}: {e}" for e in state.get("errors", []))

    async def delete_local(self, ids):
        async with self.lock:
            state = await self.local(fresh=True)
            selected = []
            for asset_id in ids:
                asset = next((a for a in state["assets"] if a["id"] == asset_id), None)
                if not asset:
                    raise ValueError("Stored files changed; refresh before deleting")
                if asset["blockers"]:
                    raise ValueError(asset["name"] + ": " + "; ".join(asset["blockers"]))
                selected.append(asset)
            # Validate the entire selection before touching any file; re-resolve server-owned paths.
            paths = await asyncio.to_thread(self._resolve_selection, selected)
            for asset in selected:
                if asset["kind"] in {"weights", "huggingface-cache"} and asset["id"] not in paths:
                    raise ValueError("Stored files changed; refresh before deleting")
            deleted, errors = [], []
            for asset in selected:
                try:
                    if asset["kind"] in {"weights", "huggingface-cache"}:
                        path, root = paths[asset["id"]]
                        if (
                            path.is_symlink()
                            or not path.resolve().is_relative_to(root)
                            or path == root
                        ):
                            raise ValueError("Unsafe storage path")
                        if path.is_dir():
                            await asyncio.to_thread(shutil.rmtree, path)
                        else:
                            await asyncio.to_thread(path.unlink)
                        for job in await asyncio.to_thread(self.store.list_downloads):
                            if (
                                job.repo_id == asset["model_id"]
                                and job.status == "completed"
                                and Path(job.target_path) == path
                            ):
                                job.status, job.message = (
                                    "deleted",
                                    "Removed from this storage location",
                                )
                                await asyncio.to_thread(self.store.save_download, job)
                        self.node.models._invalidate_inventory()
                        with self.node.models._inventory_guard:
                            self.node.models._local_fit_cache.pop(asset["model_id"], None)
                    elif asset["kind"] == "runtime-image":
                        await self.library.docker.request(
                            "DELETE",
                            "/images/" + quote(asset["path"], safe=""),
                            params={"force": "false"},
                        )
                    elif asset["kind"] == "recipe-cache":
                        await self.library.docker.request(
                            "DELETE",
                            "/volumes/" + quote(asset["path"], safe=""),
                            params={"force": "false"},
                        )
                    self.cached = None
                    await self._mark_recipe_files_removed(asset)
                    deleted.append(asset["id"])
                except (OSError, httpx.HTTPError, ValueError) as error:
                    errors.append(asset["name"] + ": " + self.node.models.safe_error(error))
            self.cached = None
            self.scan_future = None
            if self.scan_state:
                self.scan_state["assets"] = [
                    a for a in self.scan_state["assets"] if a["id"] not in deleted
                ]
            self.node.demand.profiles[:] = [
                p
                for p in self.node.demand.profiles
                if not (
                    p.id in self.node.demand.dynamic
                    and any(
                        a["kind"] == "weights" and a["model_id"] == (p.model_id or p.model)
                        for a in selected
                        if a["id"] in deleted
                    )
                )
            ]
            self.node.demand.persist()
            if self.on_change:
                self.on_change()
            return {"deleted": deleted, "errors": errors}

    async def _mark_recipe_files_removed(self, asset):
        for job in await asyncio.to_thread(self.store.runtime_objects, "recipe-install"):
            recipe = job["recipe"]
            local = self.node.local_cluster()
            if job.get("cluster_id") != (local.id if local else "controller"):
                continue
            if any(
                r["id"] == recipe["id"] and r["version"] == recipe["version"]
                for r in asset["recipe_refs"]
            ):
                job.update(artifacts_present=False)
                await asyncio.to_thread(self.store.save_runtime_object, "recipe-install", job)

    async def delete(self, ids):
        if any(":" not in value for value in ids):
            raise ValueError("Invalid storage asset identifier")
        groups = {}
        for asset_id in ids:
            cluster_id, asset = asset_id.split(":", 1)
            groups.setdefault(cluster_id, []).append(asset)
        result = {"deleted": [], "errors": []}
        local = self.node.local_cluster()
        for cluster_id, values in groups.items():
            try:
                if cluster_id == (local.id if local else "controller"):
                    done = await self.delete_local(values)
                else:
                    cluster = self.registry.get(cluster_id)
                    done = await self._remote(cluster, "POST", "/delete", {"asset_ids": values})
                result["deleted"].extend(cluster_id + ":" + value for value in done["deleted"])
                result["errors"].extend(done.get("errors", []))
            except (httpx.HTTPError, ValueError, KeyError) as error:
                result["errors"].append(self.node.models.safe_error(error))
        return result

    async def require(self, requirements, excluded_models=()):
        state = await self.inventory(fresh=True)
        shortages = []
        for location_id, needed in requirements.items():
            location = next((p for p in state["locations"] if p["id"] == location_id), None)
            if not location or location["free_bytes"] is None:
                raise ValueError("Storage capacity unavailable; refresh the selected node")
            if needed > location["free_bytes"]:
                shortages.append(
                    {
                        "location_id": location_id,
                        "name": location["cluster_name"] + " · " + location["name"],
                        "required_bytes": needed,
                        "available_bytes": location["free_bytes"],
                        "missing_bytes": needed - location["free_bytes"],
                    }
                )
        if shortages:
            raise StorageShortage(shortages, excluded_models)
        return state

    async def download_check(self, request, plan, target):
        local = await self.local(fresh=True)
        cluster = self.node.local_cluster()
        local_id = (cluster.id if cluster else "controller") + ":" + local["model_location_id"]
        installed = any(
            a["model_id"] == request.repo_id and a.get("revision") == request.revision
            for a in local["assets"]
            if a["kind"] == "weights"
        )
        requirements = {local_id: (0 if installed else plan.repository_bytes) + 64 * 1024**2}
        if target and target.kind == "remote-bridge":
            remote = await self._remote(target, "GET", "")
            present = any(
                a["model_id"] == request.repo_id and a.get("revision") == request.revision
                for a in remote["assets"]
                if a["kind"] == "weights"
            )
            requirements[target.id + ":" + remote["model_location_id"]] = (
                0 if present else plan.repository_bytes
            )
        await self.require(requirements, [request.repo_id])

    async def local_launch_check(self, model_id):
        state = await asyncio.to_thread(self._quick_disk_state)
        disk = next(p for p in state["locations"] if p["id"] == state["model_location_id"])
        reserve = 64 * 1024**2
        if disk["free_bytes"] < reserve:
            cluster = self.node.local_cluster()
            location = (cluster.id if cluster else "controller") + ":" + disk["id"]
            raise StorageShortage(
                [
                    {
                        "location_id": location,
                        "name": "Model disk",
                        "required_bytes": reserve,
                        "available_bytes": disk["free_bytes"],
                        "missing_bytes": reserve - disk["free_bytes"],
                    }
                ],
                [model_id],
            )

    async def launch_check(self, request):
        state = await self.inventory(fresh=True)
        locations = {p["id"]: p for p in state["locations"]}
        candidates = [
            a
            for a in state["assets"]
            if a.get("model_id") == request.model_id
            and (not request.cluster_id or a["cluster_id"] == request.cluster_id)
        ]
        if not candidates:
            return  # Existing launch admission explains missing weights/runtime compatibility.
        # Startup creates small metadata/cache files even when all weights are already present.
        reserve = 64 * 1024**2
        for asset in candidates:
            location = locations[asset["location_id"]]
            if location["free_bytes"] is not None and location["free_bytes"] >= reserve:
                return
        await self.require({candidates[0]["location_id"]: reserve}, [request.model_id])
