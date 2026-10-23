"""LAN discovery, pinned SSH trust and durable, idempotent cluster provisioning."""

import asyncio
import base64
import contextlib
import hashlib
import ipaddress
import json
import os
import shlex
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings
from .setup_models import (
    DiscoveryRequest,
    SetupCheck,
    SetupJob,
    SetupNode,
    SetupRequest,
    SetupStep,
    private_address,
)
from .store import SnapshotStore

STEPS = [
    ("preflight", "Check machines & fabric"),
    ("dependencies", "Prepare NVIDIA runtime & storage"),
    ("head", "Create Kubernetes head"),
    ("workers", "Join selected workers"),
    ("gpu", "Enable GPU scheduling"),
    ("storage", "Connect shared model storage"),
    ("dashboard", "Install dashboard & runtime controls"),
    ("verify", "Verify cluster readiness"),
]
SYNC_GUIDE = "https://docs.nvidia.com/sync/latest/cluster-assistant.html"


class SetupFailure(Exception):
    pass


async def process(args: list[str], data: bytes | None = None, timeout: float = 30) -> bytes:
    """Bound output and kill process groups on cancellation or timeouts."""
    proc = await asyncio.create_subprocess_exec(
        *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
    )

    async def communicate():
        if data:
            proc.stdin.write(data)
            await proc.stdin.drain()
        proc.stdin.close()
        output = bytearray()
        while chunk := await proc.stdout.read(65536):
            output.extend(chunk)
            if len(output) > 4 * 1024 * 1024:
                raise SetupFailure("Remote response exceeded its output limit")
        await proc.wait()
        if proc.returncode:
            raise SetupFailure(
                "Remote command failed. Check SSH trust, key access, sudo and node service logs."
            )
        return bytes(output)

    try:
        return await asyncio.wait_for(communicate(), timeout)
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()


def fingerprint(key: str) -> str:
    digest = hashlib.sha256(base64.b64decode(key)).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def _is_lan(address):
    try:
        private_address(address)
        return True
    except ValueError:
        return False


class SSHTransport:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.known_hosts = settings.setup_state_path / "known_hosts"

    async def host_key(self, address: str) -> tuple[str, str]:
        private_address(address)
        output = await process(["ssh-keyscan", "-T", "2", "-t", "ed25519", address], timeout=4)
        for line in output.decode().splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[1] == "ssh-ed25519":
                return parts[2], fingerprint(parts[2])
        raise SetupFailure("SSH did not provide an Ed25519 host key")

    async def trust(self, nodes: list[SetupNode]) -> None:
        # Check every key before making any SSH connection. Replace only selected hosts.
        entries = {}
        for node in nodes:
            key, actual = await self.host_key(node.address)
            if actual != node.fingerprint:
                raise SetupFailure(
                    f"Host key changed for {node.address}; verify the fingerprint on the machine"
                )
            entries[node.address] = f"{node.address} ssh-ed25519 {key}"
        self.known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        previous = self.known_hosts.read_text().splitlines() if self.known_hosts.exists() else []
        lines = [line for line in previous if line.split()[0] not in entries]
        temporary = self.known_hosts.with_suffix(".tmp")
        temporary.write_text("\n".join(lines + list(entries.values())) + "\n")
        temporary.chmod(0o600)
        temporary.replace(self.known_hosts)

    async def run(self, node: SetupNode, script: str, timeout: float = 60) -> str:
        args = [
            "ssh", "-F", str(self.settings.setup_ssh_config or "/dev/null"),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts}",
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
            "-o", "HostKeyAlgorithms=ssh-ed25519",
        ]
        if self.settings.setup_ssh_identity:
            args += ["-i", str(self.settings.setup_ssh_identity), "-o", "IdentitiesOnly=yes"]
        args += [f"{node.username}@{node.address}", "bash", "-se"]
        return (await process(args, script.encode(), timeout)).decode()

    async def upload(self, node: SetupNode, archive: bytes) -> None:
        # Archive is generated exclusively from the packaged chart; no user paths.
        encoded = base64.b64encode(archive).decode()
        await self.run(node, "mkdir -p /tmp/spark-deepseek-chart\n"
                       f"echo {shlex.quote(encoded)} | base64 -d | "
                       "tar xz -C /tmp/spark-deepseek-chart\n")


class ClusterSetupService:
    def __init__(self, settings: Settings, store: SnapshotStore, transport=None):
        self.settings = settings
        self.store = store
        self.mode = "demo" if settings.mode == "demo" else "ssh"
        self.transport = transport or SSHTransport(settings)
        self.tasks: dict[str, asyncio.Task] = {}
        self.lock = asyncio.Lock()
        self.discovery_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode == "demo" or bool(
            self.settings.setup_enabled and self.settings.model_admin_token
        )

    async def start(self):
        for job in self.store.list_setup_jobs():
            if job.status in {"queued", "running"}:
                job.status = "blocked"
                job.error = "Setup interrupted. Resume to recheck machines and reconcile each step."
                for step in job.steps:
                    if step.status == "running":
                        step.status = "failed"
                self.save(job)

    async def stop(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def save(self, job: SetupJob):
        job.updated_at = datetime.now(UTC)
        self.store.save_setup_job(job)

    def require_enabled(self):
        if not self.enabled:
            raise SetupFailure(
                "Enable SETUP_ENABLED and configure MODEL_ADMIN_TOKEN on the bootstrap dashboard"
            )

    async def state(self):
        return {
            "mode": self.mode, "enabled": self.enabled,
            "sync_guide": SYNC_GUIDE,
            "jobs": self.store.list_setup_jobs(),
            "requirements": [
                "Run this bootstrap dashboard on the same LAN as the Sparks.",
                "Verify SSH fingerprints, configure SSH key access and noninteractive sudo.",
                "Use NVIDIA Sync Cluster Assistant to configure cabled ConnectX-7 networking.",
                "Four nodes require a switch. Internet access is needed for pinned installers.",
            ],
        }

    async def discover(self, request: DiscoveryRequest):
        self.require_enabled()
        if self.discovery_lock.locked():
            raise SetupFailure("Discovery is already running")
        async with self.discovery_lock:
            return await self._discover(request)

    async def _discover(self, request: DiscoveryRequest):
        if self.mode == "demo":
            return {"nodes": [{
                "address": f"192.168.50.{10 + index}", "hostname": f"spark-{index + 1:02}",
                "source": "demo", "fingerprint": "SHA256:" + "A" * 43,
                "gpu": "Unverified until preflight", "ssh": True,
            } for index in range(4)], "notes": ["Demo discovery — no network traffic sent."]}
        candidates = {address: (address, "manual") for address in request.addresses}
        notes = []
        try:
            from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

            zeroconf = AsyncZeroconf()
            names = set()

            def found(zeroconf, service_type, name, state_change):
                names.add((service_type, name))

            browser = AsyncServiceBrowser(
                zeroconf.zeroconf, ["_ssh._tcp.local.", "_workstation._tcp.local."],
                handlers=[found],
            )
            try:
                await asyncio.sleep(3)
                for service_type, name in sorted(names)[:64]:
                    info = await zeroconf.async_get_service_info(service_type, name, timeout=250)
                    if info:
                        for address in info.parsed_addresses():
                            with contextlib.suppress(ValueError):
                                candidates[private_address(address)] = (
                                    info.server.rstrip("."), "mDNS"
                                )
            finally:
                await browser.async_cancel()
                await zeroconf.async_close()
        except (ImportError, OSError):
            notes.append("mDNS unavailable here; enter a LAN IP or a /24 subnet.")
        if request.subnet:
            for host in ipaddress.ip_network(request.subnet).hosts():
                candidates.setdefault(str(host), (str(host), "subnet"))
        semaphore = asyncio.Semaphore(24)

        async def inspect(address, metadata):
            async with semaphore:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(address, 22), 0.6
                    )
                    writer.close()
                    await writer.wait_closed()
                    _, host_fingerprint = await self.transport.host_key(address)
                    return {
                        "address": address, "hostname": metadata[0], "source": metadata[1],
                        "ssh": True, "fingerprint": host_fingerprint,
                        "gpu": "Unverified until preflight",
                    }
                except (OSError, TimeoutError, SetupFailure):
                    return None
        nodes = await asyncio.gather(*(inspect(ip, meta) for ip, meta in candidates.items()))
        return {"nodes": [node for node in nodes if node], "notes": notes + [
            "SSH discovery does not prove GB10 compatibility. Verify fingerprints on the devices."
        ]}

    async def preflight(self, request: SetupRequest) -> list[SetupCheck]:
        self.require_enabled()
        async with self.lock:
            self._no_active_job()
            return await self._preflight(request)

    async def _preflight(self, request: SetupRequest) -> list[SetupCheck]:
        if self.mode == "demo":
            return [SetupCheck(node=node.address, code="demo", status="pass",
                               message="Simulated GB10, fabric, SSH and storage checks")
                    for node in request.nodes]
        await self.transport.trust(request.nodes)
        checks = []
        facts = {}
        probe = Path(__file__).with_name("setup_probe.py").read_text()
        for node in request.nodes:
            try:
                data = json.loads(await self.transport.run(node, "python3 - <<'PROBE'\n"
                                                          + probe + "\nPROBE\n"))
                facts[node.address] = data
            except (SetupFailure, OSError, ValueError):
                checks.append(SetupCheck(
                    node=node.address, code="ssh", status="error",
                    message="SSH key authentication or Python probe failed",
                    remedy="Verify the host fingerprint and SSH key. Install Python 3 on DGX OS.",
                ))
                continue
            def check(code, passed, message, remedy, automatic=False, warning=False):
                checks.append(SetupCheck(
                    node=node.address, code=code,
                    status="pass" if passed else "warning" if warning or automatic else "error",
                    message=message, remedy="" if passed else remedy, automatic=automatic and not passed,
                ))
            check("gb10", "GB10" in data["gpu"] and data["architecture"] == "aarch64",
                  f"{data['gpu'] or 'GPU unavailable'} · {data['architecture']}",
                  "Use a DGX Spark / GB10 running current DGX OS and a working NVIDIA driver.")
            check("memory", data["memory_gib"] >= 100,
                  f"{data['memory_gib']:.0f} GiB unified memory", "Use a 128 GiB GB10 system.")
            check("disk", data["disk_gib"] >= 50,
                  f"{data['disk_gib']:.0f} GiB free on /var/lib",
                  "Free at least 50 GiB for base services; models need additional space.")
            check("sudo", data["sudo"], "Noninteractive sudo access",
                  "Configure a bootstrap account with sudo -n access, then retry. No passwords are stored.")
            check("toolkit", data["toolkit"], "NVIDIA container runtime",
                  "Install nvidia-container-toolkit from the DGX OS package repositories.", True)
            check("ownership", not data["k3s"] or (data["managed_cluster"] == request.name
                  and data.get("managed_role") == node.role),
                  "Cluster ownership", "An existing unmanaged k3s installation was found. "
                  "Use dedicated nodes or administer that cluster separately.")
            check("clock", abs(time.time() - data["clock"]) < 60,
                  "System clock is synchronized", "Enable NTP and correct the system clock.")
            if len(request.nodes) > 1:
                if not node.fabric_address:
                    addresses = [address for iface in data["interfaces"]
                                 if iface["up"] and iface["speed_mbps"] >= 200000
                                 for address in iface["addresses"]]
                    addresses = [address for address in addresses if _is_lan(address)]
                    if len(addresses) == 1:
                        node.fabric_address = addresses[0]
                selected = node.fabric_address
                active = [iface for iface in data["interfaces"] if iface["up"] and
                          iface["speed_mbps"] >= 200000 and selected in iface["addresses"]]
                check("fabric", bool(selected and active and data["rdma"]),
                      "ConnectX-7 address, 200 Gbit/s link and RDMA device",
                      "Run NVIDIA Sync > Settings > Cluster Assistant, connect supported QSFP cables, "
                      "then enter each node's configured fabric IPv4 address.")
        hostnames = [data["hostname"] for data in facts.values()]
        if len(hostnames) != len(set(hostnames)):
            checks.append(SetupCheck(node="cluster", code="hostname", status="error",
                                     message="Duplicate hostnames", remedy="Rename nodes uniquely."))
        releases = {data["dgx_release"] for data in facts.values() if data.get("dgx_release")}
        if len(releases) > 1:
            checks.append(SetupCheck(
                node="cluster", code="os-release", status="error", message="DGX OS releases differ",
                remedy="Update every node to the same current DGX OS release in DGX Dashboard.",
            ))
        drivers = {data["driver"] for data in facts.values()}
        if len(drivers) > 1:
            checks.append(SetupCheck(
                node="cluster", code="driver", status="error", message="NVIDIA drivers differ",
                remedy="Update all nodes to the same DGX OS / driver release using DGX Dashboard.",
            ))
        if not any(check.status == "error" for check in checks) and len(request.nodes) > 1:
            for node in request.nodes:
                peers = [peer.fabric_address for peer in request.nodes if peer != node]
                command = "\n".join(f"ping -I {shlex.quote(node.fabric_address)} -c 2 -W 2 "
                                    f"{shlex.quote(peer)} >/dev/null" for peer in peers)
                try:
                    await self.transport.run(node, command, timeout=30)
                    checks.append(SetupCheck(node=node.address, code="fabric-reachability",
                                             status="pass", message="All peer fabric addresses reachable"))
                except SetupFailure:
                    checks.append(SetupCheck(
                        node=node.address, code="fabric-reachability", status="error",
                        message="A peer fabric address is unreachable",
                        remedy="Check Sync routes, QSFP cabling, switch ports and ICMP policy.",
                    ))
        return checks

    async def launch(self, request: SetupRequest) -> SetupJob:
        self.require_enabled()
        async with self.lock:
            self._no_active_job()
            job = SetupJob(request=request, mode=self.mode, steps=[
                SetupStep(id=key, label=label) for key, label in STEPS
            ])
            self.save(job)
            self.tasks[job.id] = asyncio.create_task(self._run(job))
            return job.model_copy(deep=True)

    def _no_active_job(self):
        if any(job.status in {"running", "queued"} for job in self.store.list_setup_jobs()):
            raise SetupFailure("Another cluster setup is in progress")

    async def resume(self, job_id: str) -> SetupJob:
        self.require_enabled()
        async with self.lock:
            job = self.store.get_setup_job(job_id)
            if job is None:
                raise KeyError(job_id)
            self._no_active_job()
            if job.mode != self.mode or job.status not in {"blocked", "failed"}:
                raise SetupFailure("Only an interrupted or failed setup in the same mode can resume")
            # Reconcile every idempotent step; never trust persisted completion as live evidence.
            for step in job.steps:
                step.status = "pending"
            job.status, job.error = "queued", None
            self.save(job)
            self.tasks[job.id] = asyncio.create_task(self._run(job))
            return job.model_copy(deep=True)

    def log(self, job, message, node="cluster"):
        job.logs.append({"timestamp": datetime.now(UTC).isoformat(),
                         "node": node, "message": message})
        job.logs = job.logs[-500:]
        self.save(job)

    async def _run(self, job):
        current = None
        try:
            job.status = "running"
            self.save(job)
            for current in job.steps:
                current.status = "running"
                self.log(job, current.label)
                if current.id == "preflight":
                    job.checks = await self._preflight(job.request)
                    if any(check.status == "error" for check in job.checks):
                        job.status = "blocked"
                        raise SetupFailure("Resolve the highlighted diagnostics, then resume setup")
                elif self.mode == "demo":
                    await asyncio.sleep(0.15)
                else:
                    await self._apply(job, current.id)
                current.status = "done"
                current.message = "Simulated" if self.mode == "demo" else "Verified"
                self.save(job)
            job.status = "ready"
            head = next(node for node in job.request.nodes if node.role == "head")
            job.dashboard_url = ("#models" if self.mode == "demo" else
                                 f"http://{head.address}:30080")
            self.log(job, "Cluster services are ready. Add a model to validate inference and NCCL.")
        except asyncio.CancelledError:
            job.status = "blocked"
            job.error = "Setup interrupted. Resume to reconcile node state."
            if current:
                current.status = "failed"
            self.save(job)
            raise
        except Exception as error:
            if job.status != "blocked":
                job.status = "failed"
            if current:
                current.status = "failed"
            # Transport output is never persisted; it may include credentials.
            job.error = str(error) if isinstance(error, SetupFailure) else (
                "Setup could not complete. Check node connectivity and retry this step."
            )
            if current and job.status == "failed":
                remedies = {
                    "dependencies": "Check apt repository access and sudo. Retry installs the missing packages.",
                    "head": "Check sudo journalctl -u k3s and access to get.k3s.io. Retry reconciles k3s.",
                    "workers": "Check TCP 6443 to the head and sudo journalctl -u k3s-agent on workers.",
                    "gpu": "Check NVIDIA container runtime and the device-plugin logs in kube-system.",
                    "storage": "Check NFS service, export permissions and TCP 2049 between selected nodes.",
                    "dashboard": "Check image pull access, PVC binding and pods/events in inference.",
                    "verify": "Check dashboard/agent pod readiness, NodePort 30080 and NFS mounts.",
                }
                job.checks.append(SetupCheck(
                    node="cluster", code=current.id, status="error", message=job.error,
                    remedy=remedies.get(current.id, "Check SSH access and retry."), automatic=True,
                ))
            self.log(job, job.error)
        finally:
            self.tasks.pop(job.id, None)

    async def _apply(self, job, step):
        from .setup_scripts import apply_step
        await apply_step(self, job, step)
