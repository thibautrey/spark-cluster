# Guided cluster setup

Open **Cluster setup** in the dashboard to discover machines, assign roles, check readiness, install
cluster services and follow progress. Demo mode runs the entire wizard with simulated machines and
never sends discovery probes or SSH commands.

## Start a bootstrap dashboard

Run the bootstrap dashboard on a trusted machine on the same management LAN as the Sparks. It needs
Python 3.11+, OpenSSH (`ssh` and `ssh-keyscan`), network access to the selected machines and an existing
SSH key that those machines authorize. The Python package includes the deployment chart.

```bash
python -m pip install -e .
export SPARK_DEEPSEEK_MODE=live
export SPARK_DEEPSEEK_SETUP_ENABLED=true
export SPARK_DEEPSEEK_SETUP_SSH_IDENTITY=/absolute/path/to/existing-key
# Set SPARK_DEEPSEEK_MODEL_ADMIN_TOKEN through your secret manager or environment.
spark-deepseek serve --host 127.0.0.1 --port 8080
```

Enter the same administration token in the wizard and click **Connect**. Live discovery and setup
require both explicit enablement and a configured administration token. Keep the bootstrap dashboard
bound to loopback, or place it behind an authenticated TLS proxy. The token stays in the page's memory;
it is not written to browser storage. A local SSH agent can be used instead of `SETUP_SSH_IDENTITY`.
`SETUP_SSH_CONFIG` optionally points to a dedicated client configuration; the default ignores the
user's global SSH configuration. Containers need the existing key or agent socket mounted explicitly,
a writable `SETUP_STATE_PATH`, and host networking for LAN multicast discovery on Linux. Native
execution is recommended for bootstrap, especially on macOS where container multicast is isolated.

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
5. Create shared model storage at `/var/lib/spark-deepseek/models` on the head. NFS exports allow only
   selected cluster node addresses and preserve root squashing. A retained static RWX volume binds to
   the model PVC. Its advertised 1 TiB is a scheduling capacity, not a disk reservation or quota;
   model downloads still check actual free space.
6. Verify Helm `v3.19.0` against its published SHA256, then install the bundled chart, telemetry agents,
   dashboard, admin-token secret and idle runtime resources. The image must be accessible from every
   node. Configure `SETUP_IMAGE_REPOSITORY` and `SETUP_IMAGE_TAG` for your published build.
7. Check GPU registration, deployment/agent readiness, PVC binding, dashboard HTTP health and actual
   NFS read/write access from each selected machine as the application UID.

Open the resulting dashboard on the head at port **30080**. Its administration token is the same as
the bootstrap token. Add/download a model in the model library, then use **Launch**. Model capacity
and runtime readiness are validated separately. Launching scales the deferred head and required Ray
workers; a successful setup does not mean an inference model is already running.

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
| `POST /api/v1/setup/discover` | mDNS discovery and optional bounded LAN probes |
| `POST /api/v1/setup/check` | Read-only preflight and detected node fabric addresses |
| `POST /api/v1/setup/jobs` | Start a reviewed installation |
| `POST /api/v1/setup/jobs/{id}/resume` | Revalidate and resume an interrupted installation |

The dashboard polls authenticated setup state every two seconds while a job is running. Logs and
step state survive browser reloads. The state directory and SQLite database must be on persistent
storage for restart recovery.
