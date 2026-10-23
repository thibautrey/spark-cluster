# Guided cluster setup

Run the dashboard on any Linux, macOS, NAS, VM, or container host that can reach the Spark management
LAN. The interface remains on that system. The controller downloads model files from the internet
once, while a lightweight bridge inside Kubernetes copies them over the LAN into shared model storage
and performs log collection and runtime rollouts next to the GPUs. The controller relays
the bridge APIs and WebSockets, so cache, memory, topology, models, launches and logs all remain in one
dashboard. Demo mode runs the same interface with simulated machines and sends no LAN or SSH traffic.

## Start a bootstrap dashboard

Run the bootstrap dashboard on a trusted machine on the same management LAN as the Sparks. It needs
Python 3.11+, OpenSSH (`ssh` and `ssh-keyscan`), network access to the selected machines and an existing
SSH key that those machines authorize. The Python package includes the deployment chart.

```bash
python -m pip install -e .
export SPARK_CLUSTER_MODE=live
export SPARK_CLUSTER_SETUP_ENABLED=true
export SPARK_CLUSTER_SETUP_SSH_IDENTITY=/absolute/path/to/existing-key
# Set SPARK_CLUSTER_MODEL_ADMIN_TOKEN through your secret manager or environment.
spark-cluster serve --host 0.0.0.0 --port 8080
```

Enter the same administration token in the wizard and click **Connect**. Live discovery and setup
require both explicit enablement and a configured administration token. Limit port 8080 to trusted
management clients, or place it behind an authenticated TLS proxy. The token stays in the page's memory;
it is not written to browser storage. A local SSH agent can be used instead of `SETUP_SSH_IDENTITY`.
`SETUP_SSH_CONFIG` optionally points to a dedicated client configuration; the default ignores the
user's global SSH configuration. Containers need the existing key or agent socket mounted explicitly,
a writable `SETUP_STATE_PATH`, and host networking for LAN multicast discovery on Linux. Native
execution is recommended for bootstrap, especially on macOS where container multicast is isolated.

To attach a bridge that is already installed, enter its head-node management IPv4 and NodePort in
**Already configured**. No SSH access is needed for attachment. The controller authenticates
`/api/v1/controller/verify`, stores only the bridge URL and cluster identity in
`SETUP_STATE_PATH/active-cluster.json` with mode `0600`, then immediately begins relaying cluster data.
Set `SPARK_CLUSTER_REMOTE_CLUSTER_URL=http://192.168.1.42:30080` for an explicitly managed connection
without the UI. Public, loopback, link-local and URL-with-path targets are rejected.

The controller and bridge share `MODEL_ADMIN_TOKEN`, but the token is never sent across the LAN.
Mutating requests and attachment checks use an HMAC-SHA256 signature over method, target, timestamp and
body digest. Signatures expire after 30 seconds and prevent replay or modification. Keep clocks synced.
Read-only telemetry remains available on the private bridge endpoint; restrict NodePort 30080 to the
management LAN with host or upstream firewall rules.

### Controller-first model delivery

When a cluster is attached, model downloads stay on the external controller:

1. The controller resolves Hugging Face metadata and verifies fit against a fresh bridge snapshot.
2. Hugging Face files download once into `MODEL_CACHE_PATH` on the controller with normal resume.
3. The controller hashes every completed file and stores a signed transfer manifest.
4. The bridge opens authenticated, range-capable HTTP requests back to the controller and resumes
   partial files after interruptions.
5. Each SHA-256 digest is verified before the staged directory is atomically installed into the
   shared model volume. Runtime nodes mount that volume read-only and do not contact Hugging Face.

Set `SPARK_CLUSTER_CONTROLLER_PUBLIC_URL` to an HTTP(S) URL using the controller's reachable private
IPv4 address, for example `http://192.168.1.149:8769`. It may be omitted when the browser reaches the
dashboard through that same LAN address. The URL is rejected unless it targets an RFC1918 address.
Allow the bridge to initiate connections to this port. File and manifest requests require the shared
HMAC secret, file paths are restricted to the staged model, and all content is verified against the
signed manifest. A failed LAN copy can be retried without another internet download.

The SSH user must have noninteractive `sudo -n` access on the selected devices. Prepare that account
according to your administration policy. The wizard does not collect SSH passwords, sudo passwords or
private keys. Confirm each displayed host fingerprint against the device itself:

```bash
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Only explicitly selected, fingerprint-pinned machines are accessed. A changed key blocks setup.
Verified public host keys are stored under `SETUP_STATE_PATH/known_hosts`, independently of your normal
SSH trust file. Never bypass the mismatch by trusting an unverified replacement.

## Discovery and fabric

Discovery listens for `_ssh._tcp.local.` and `_workstation._tcp.local.` mDNS services. If the network
suppresses multicast or the device does not advertise these services, enter its RFC1918 IPv4 address,
or opt into a single `/24` (or smaller) subnet search. Probes only check SSH on port 22, with bounded
concurrency and timeouts. Reachable SSH hosts remain unverified candidates until the GB10 preflight.
Public addresses, link-local addresses and wider subnet scans are rejected.

For multiple devices, connect the QSFP fabric and use the official
[NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html). It configures
ConnectX-7 networking and inter-device SSH. It supports 2–4 DGX Spark / GB10 systems; four devices
require a switch. Direct and switched links must not be mixed. Update systems to the current DGX OS
release first. See [direct connections](https://docs.nvidia.com/sync/latest/direct-connections.html)
for NVIDIA Sync device discovery and account setup.

NVIDIA documents a desktop workflow rather than a public cluster-configuration API. The wizard links
to that workflow and independently checks the resulting configuration. A single active 200 Gbit/s
fabric IPv4 address is detected automatically. When several exist, enter the appropriate address from
Sync for each node. Preflight verifies link state, RDMA-device presence and all-pairs ICMP connectivity
using the selected source addresses. These checks establish reachability, not measured bandwidth or
NCCL performance. Use Sync's link-speed test and a model launch for those checks. One Spark needs no
fabric step.

### Visual cable assistant

After selecting head and worker roles, the dashboard immediately draws the expected physical layout:

- two Sparks: one direct cable;
- three Sparks with direct links: three cables forming a triangle, with every Spark connected to the
  other two;
- two to four Sparks through a switch: one cable from every Spark to the same switch;
- one Spark: management Ethernet and power only.

Select **Inspect cables** after verifying SSH fingerprints. The read-only probe maps each Linux network
interface to its PCI device and `mlx5`/RDMA device, then reads carrier state, negotiated speed, MTU,
MAC address, IP prefix, RDMA port state and neighbor table. A bounded second pass sends interface-bound
pings to selected peers on each fabric subnet and checks their neighbor MAC against the expected
remote interface. Direct layouts also require point-to-point /30 or /31 subnets; a subnet match alone
stays amber. Switch layouts require one active link from every selected node with verified peers.
Links with verified peers at or above the documented 184 Gbit/s lower bound appear green; slower or
unverified carrier appears amber; missing expected links
appear red and block installation.

The result names the affected Sparks and the detected interface or available ConnectX-7 port. The
diagram then gives a numbered instruction such as connecting one node's port to a particular peer or
to the switch. Reseat the cable, wait for the port LEDs, and run inspection again. The assistant does
not infer a healthy cable from interface presence alone, and the complete preflight repeats the same
physical-topology gate before any installation changes.

The illustrated guide distinguishes RJ45 management Ethernet from the wide QSFP fabric connectors.
Port drawings are schematic; Linux interface names do not establish left/right physical position.
Carrier and verified peer connectivity do not reveal the exact physical cable path or switch chassis.
An unreachable machine may be powered off, disconnected, or blocked by a network policy; the dashboard
does not claim to distinguish those conditions without telemetry.

## What installation does

The review page lists exact machines, users and roles before applying changes. Steps are persisted in
SQLite; bounded, structured progress logs contain operation summaries rather than remote stdout.
Enrollment and administration tokens are passed through pinned SSH standard input and excluded from
dashboard state and logs. The remote cluster retains its own Kubernetes credentials as required by
k3s; temporary enrollment files are root-only and removed on exit.

1. Validate GB10/ARM64, unified memory, free storage, SSH, sudo, matching drivers/DGX OS releases,
   clock skew, ownership and fabric connectivity.
2. Install the NVIDIA container toolkit and NFS client packages from the configured DGX OS apt
   repositories. Existing NVIDIA drivers and DGX OS versions are not replaced.
3. Install k3s `v1.34.1+k3s1` on the head and join the selected workers. The official installer verifies
   its versioned binary. Unmanaged k3s installations are rejected.
4. Deploy NVIDIA's device plugin `v0.17.4` and verify an allocatable GPU on every selected node.
5. Create shared model storage at `/var/lib/spark-cluster/models` on the head. NFS exports allow only
   selected cluster node addresses and preserve root squashing. A retained static RWX volume binds to
   the model PVC. Its advertised 1 TiB is a scheduling capacity, not a disk reservation or quota;
   model downloads still check actual free space.
6. Verify Helm `v3.19.0` against its published SHA256, then install the bundled chart, telemetry agents,
   authenticated cluster bridge and idle runtime resources. The image must be accessible from every
   node. Configure `SETUP_IMAGE_REPOSITORY` and `SETUP_IMAGE_TAG` for your published build.
7. Check GPU registration, deployment/agent readiness, PVC binding, dashboard HTTP health and actual
   NFS read/write access from each selected machine as the application UID.

After verification, the current external dashboard automatically attaches to the bridge on head port
**30080**. Add/download a model in the model library, then use **Launch**. The controller stages the
model locally and the bridge copies it into the cluster NFS volume before exposing it to the runtime;
the external controller does not need that volume mounted. Model capacity and runtime readiness are
validated separately. Launching scales the
deferred head and required Ray workers. The OpenAI-compatible runtime is exposed on head NodePort
**30000**, and the API panel displays that LAN URL after launch.

For an image hosted by an HTTP registry on the trusted LAN, set `SETUP_REGISTRY_ENDPOINT` to its origin
and `SETUP_IMAGE_REPOSITORY` to a repository under the same host and port. The assistant writes the
matching k3s mirror configuration on every selected node before installation. Set
`SETUP_IMAGE_DIGEST=sha256:...` to pin the exact manifest; the generated Kubernetes workloads then use
`repository@digest`, so a mutable or replaced registry tag cannot change the deployed image. Registry
write access should remain restricted to administrators even when every deployment is digest-pinned.

The head is a single control-plane and storage server. Keep API, Ray, NFS and dashboard ports on a
trusted private network. Allow SSH 22, Kubernetes 6443/10250, k3s overlay 8472/UDP, NFS 2049 and
NodePort 30080 where needed; Ray/NCCL host-network traffic must be reachable among fabric peers.
The wizard does not disable firewalls or change switch settings. TLS and ingress authentication for
remote users remain an operator configuration.

## Recovery and assisted fixes

Checks show the affected machine, evidence, severity and a concrete remedy. Missing toolkit packages
are marked for automatic repair during installation. SSH access, unsupported hardware, mismatched
software, clock drift, cabling and ownership problems block changes until corrected. Failures during
installation include step-specific service, port, storage or image-pull troubleshooting guidance.

After a dashboard restart, an interrupted job becomes **blocked**, retaining its original node plan and
step logs. Select **Retry / resume setup** to recheck SSH pins, rerun preflight and reconcile the
idempotent operations. It never resumes unattended after a restart. Use **Revise plan** when addresses,
accounts or roles need correction. Only one installation runs at a time. Do not change head/worker
roles on an already configured cluster; use dedicated nodes for a new cluster. Storage and existing
models are retained on retries. There is no automatic destructive uninstall.

## API

All setup endpoints use the model administration bearer token in live mode.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/setup` | Mode, enablement, saved jobs and progress logs |
| `POST /api/v1/setup/connect` | Authenticate and persist an existing LAN bridge |
| `POST /api/v1/setup/discover` | mDNS discovery and optional bounded LAN probes |
| `POST /api/v1/setup/hardware` | Inspect ports and build the physical cable diagram |
| `POST /api/v1/setup/check` | Read-only preflight and detected node fabric addresses |
| `POST /api/v1/setup/jobs` | Start a reviewed installation |
| `POST /api/v1/setup/jobs/{id}/resume` | Revalidate and resume an interrupted installation |
| `POST /api/v1/models/import` | Start or resume a controller-to-cluster model copy |
| `GET /api/v1/downloads/{id}` | Read internet or LAN transfer progress |
| `GET /api/v1/controller/downloads/{id}/manifest` | Retrieve the signed model file manifest |
| `GET /api/v1/controller/downloads/{id}/files/{path}` | Resume an authenticated model file transfer |
| `GET /api/v1/controller/verify` | HMAC-authenticated bridge identity check |

The dashboard polls authenticated setup state every two seconds while a job is running. Logs and
step state survive browser reloads. The state directory and SQLite database must be on persistent
storage for restart recovery.
