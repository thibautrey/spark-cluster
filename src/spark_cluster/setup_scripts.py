"""Fixed bootstrap operations. Inputs are validated contracts and shell-quoted."""

import base64
import io
import json
import re
import shlex
import tarfile
from pathlib import Path
from urllib.parse import urlparse

from .cluster_setup import SetupFailure
from .setup_models import private_address

K3S_VERSION = "v1.34.1+k3s1"
HELM_VERSION = "v3.19.0"
GPU_PLUGIN_VERSION = "v0.17.4"


def chart_archive() -> bytes:
    chart = Path(__file__).parent / "chart"
    if not (chart / "Chart.yaml").exists():
        raise SetupFailure("The packaged Helm chart is unavailable")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(chart, arcname="spark-cluster")
    return output.getvalue()


def cluster_ip(node):
    return node.fabric_address or node.address


def registry_configuration(settings) -> str:
    endpoint = settings.setup_registry_endpoint
    if not endpoint:
        return ""
    parsed = urlparse(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise SetupFailure("SETUP_REGISTRY_ENDPOINT must be an HTTP(S) LAN origin")
    try:
        private_address(parsed.hostname)
        port = parsed.port
    except ValueError as error:
        raise SetupFailure("SETUP_REGISTRY_ENDPOINT must use a private LAN IPv4 address") from error
    repository_prefix = f"{parsed.hostname}:{port}" if port else parsed.hostname
    if not settings.setup_image_repository.startswith(repository_prefix + "/"):
        raise SetupFailure("SETUP_IMAGE_REPOSITORY must belong to the configured LAN registry")
    registry = (
        f'mirrors:\n  "{repository_prefix}":\n    endpoint:\n      - "{endpoint.rstrip("/")}"\n'
    )
    encoded = base64.b64encode(registry.encode()).decode()
    return (
        "sudo -n install -d -m 755 /etc/rancher/k3s\n"
        f"echo {shlex.quote(encoded)} | base64 -d | sudo -n tee "
        "/etc/rancher/k3s/registries.yaml >/dev/null\n"
        "if sudo -n systemctl is-active --quiet k3s; then "
        "sudo -n systemctl restart k3s; "
        "elif sudo -n systemctl is-active --quiet k3s-agent; then "
        "sudo -n systemctl restart k3s-agent; fi\n"
    )


def guard(name, role):
    return (
        "if command -v k3s >/dev/null; then\n"
        f'  test "$(cat /var/lib/spark-cluster/cluster-name 2>/dev/null)" = {shlex.quote(name)}\n'
        f'  test "$(cat /var/lib/spark-cluster/cluster-role 2>/dev/null)" = {shlex.quote(role)}\n'
        "fi\n"
    )


def install_k3s(arguments):
    # Official installer, explicit binary version, bounded curl, no shell argv secrets.
    return (
        "installer=$(mktemp)\ntrap 'rm -f \"$installer\"' EXIT\n"
        "curl -fsSL --connect-timeout 10 --max-time 120 https://get.k3s.io "
        '-o "$installer"\n'
        f"sudo -n env INSTALL_K3S_VERSION={shlex.quote(K3S_VERSION)} "
        f"INSTALL_K3S_EXEC={shlex.quote(arguments)} "
        'sh "$installer" >/dev/null 2>&1\n'
    )


async def apply_step(service, job, step):
    nodes = job.request.nodes
    head = next(node for node in nodes if node.role == "head")
    name = job.request.name
    run = service.transport.run
    if step == "dependencies":
        for node in nodes:
            script = (
                guard(name, node.role)
                + """
sudo -n apt-get update -qq >/dev/null 2>&1
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  curl ca-certificates tar nfs-common nvidia-container-toolkit >/dev/null 2>&1
command -v nvidia-container-runtime >/dev/null
sudo -n mkdir -p /var/lib/spark-cluster
"""
                + registry_configuration(service.settings)
            )
            await run(node, script, timeout=600)
            service.log(job, "NVIDIA container toolkit and NFS client ready", node.address)
    elif step == "head":
        ip = cluster_ip(head)
        # Marker is written before installation; partial installation can be reconciled.
        script = (
            guard(name, head.role)
            + (
                f"echo {shlex.quote(name)} | sudo -n tee "
                "/var/lib/spark-cluster/cluster-name >/dev/null\n"
                "echo head | sudo -n tee /var/lib/spark-cluster/cluster-role >/dev/null\n"
            )
            + install_k3s(
                f"server --node-ip={ip} --advertise-address={ip} --tls-san={head.address} "
                "--disable=traefik --disable=servicelb --default-runtime=nvidia "
                "--write-kubeconfig-mode=600"
            )
            + "ready=0\nfor attempt in $(seq 1 30); do\n"
            "  if sudo -n k3s kubectl wait --for=condition=Ready nodes "
            "--all --timeout=10s >/dev/null 2>&1; then ready=1; break; fi\n"
            '  sleep 2\ndone\ntest "$ready" = 1\n'
        )
        await run(head, script, timeout=600)
    elif step == "workers":
        # Join token stays in process memory and travels over pinned SSH stdin only.
        token = (await run(head, "sudo -n cat /var/lib/rancher/k3s/server/node-token\n")).strip()
        if not token or any(character.isspace() for character in token):
            raise SetupFailure("The head did not return a valid join token")
        try:
            for node in nodes:
                if node.role == "head":
                    continue
                arguments = f"agent --node-ip={cluster_ip(node)} --default-runtime=nvidia"
                script = guard(name, node.role) + (
                    f"echo {shlex.quote(name)} | sudo -n tee "
                    "/var/lib/spark-cluster/cluster-name >/dev/null\n"
                    "echo worker | sudo -n tee /var/lib/spark-cluster/cluster-role >/dev/null\n"
                    "sudo -n install -m 600 /dev/null /var/lib/spark-cluster/join.env\n"
                    f"printf '%s\\n' "
                    f"{shlex.quote('K3S_URL=https://' + cluster_ip(head) + ':6443')} "
                    f"{shlex.quote('K3S_TOKEN=' + shlex.quote(token))} | "
                    "sudo -n tee /var/lib/spark-cluster/join.env >/dev/null\n"
                    "installer=$(mktemp)\n"
                    "trap 'sudo -n rm -f /var/lib/spark-cluster/join.env; "
                    'rm -f "$installer"\' EXIT\n'
                    "curl -fsSL --connect-timeout 10 --max-time 120 https://get.k3s.io "
                    '-o "$installer"\n'
                )
                # Quote the complete root shell body once to preserve INSTALL_K3S_EXEC spaces.
                root = (
                    "set -a; . /var/lib/spark-cluster/join.env; "
                    f"INSTALL_K3S_VERSION={shlex.quote(K3S_VERSION)} "
                    f"INSTALL_K3S_EXEC={shlex.quote(arguments)} "
                    'sh "$1"'
                )
                script += (
                    f'sudo -n bash -c {shlex.quote(root)} setup "$installer" >/dev/null 2>&1\n'
                )
                await run(node, script, timeout=600)
                service.log(job, "Joined cluster using a private enrollment token", node.address)
        finally:
            token = ""
    elif step == "gpu":
        script = f"""
sudo -n k3s kubectl wait --for=condition=Ready nodes --all --timeout=240s >/dev/null
plugin_url=https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin
plugin_url=$plugin_url/{GPU_PLUGIN_VERSION}/deployments/static/nvidia-device-plugin.yml
sudo -n k3s kubectl apply -f "$plugin_url" >/dev/null
sudo -n k3s kubectl -n kube-system rollout status \
  daemonset/nvidia-device-plugin-daemonset --timeout=240s >/dev/null
for attempt in $(seq 1 60); do
  ready=$(sudo -n k3s kubectl get nodes -o json | python3 -c \
    'import json,sys
nodes=json.load(sys.stdin)["items"]
print(int(bool(nodes) and all(
    int(n["status"].get("allocatable",{{}}).get("nvidia.com/gpu",0)) > 0 for n in nodes)))')
  test "$ready" = 1 && break
  sleep 2
done
sudo -n k3s kubectl get nodes -o json
"""
        result = json.loads(await run(head, script, timeout=360))
        expected = {cluster_ip(node) for node in nodes}
        actual = set()
        all_ips = {
            a["address"]
            for item in result["items"]
            for a in item["status"]["addresses"]
            if a["type"] == "InternalIP"
        }
        if all_ips != expected:
            raise SetupFailure(
                "Cluster membership differs from the selected machines. "
                "Review existing k3s nodes before continuing."
            )
        for item in result["items"]:
            ips = {a["address"] for a in item["status"]["addresses"] if a["type"] == "InternalIP"}
            if ips & expected:
                if int(item["status"].get("allocatable", {}).get("nvidia.com/gpu", 0)) < 1:
                    raise SetupFailure(
                        "GPU resources are missing. Check NVIDIA toolkit and k3s logs."
                    )
                actual |= ips & expected
                await run(
                    head,
                    f"sudo -n k3s kubectl label node {shlex.quote(item['metadata']['name'])} "
                    "spark-cluster.io/managed=true --overwrite >/dev/null\n",
                )
        if actual != expected:
            raise SetupFailure("Not all selected nodes joined with an allocatable GPU")
    elif step == "storage":
        clients = " ".join(
            f"{cluster_ip(node)}(rw,sync,no_subtree_check,root_squash)" for node in nodes
        )
        export = f"/var/lib/spark-cluster/models {clients}\n"
        encoded = base64.b64encode(export.encode()).decode()
        await run(
            head,
            """
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-kernel-server >/dev/null 2>&1
sudo -n install -d -o 10001 -g 10001 -m 0770 /var/lib/spark-cluster/models
sudo -n install -d -o root -g root -m 0755 /etc/exports.d
"""
            + f"echo {shlex.quote(encoded)} | base64 -d | sudo -n tee "
            "/etc/exports.d/spark-cluster.exports >/dev/null\n"
            "sudo -n exportfs -ra\nsudo -n systemctl enable --now nfs-server >/dev/null\n",
            timeout=600,
        )
        manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolume",
            "metadata": {
                "name": "spark-cluster-models",
                "labels": {"spark-cluster.io/storage": name},
            },
            "spec": {
                "capacity": {"storage": "1Ti"},
                "accessModes": ["ReadWriteMany"],
                "persistentVolumeReclaimPolicy": "Retain",
                "storageClassName": "",
                "mountOptions": ["nfsvers=4.2"],
                "nfs": {"server": cluster_ip(head), "path": "/var/lib/spark-cluster/models"},
            },
        }
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "spark-cluster-models", "namespace": "inference"},
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": "",
                "volumeName": "spark-cluster-models",
                "resources": {"requests": {"storage": "1Ti"}},
            },
        }
        await run(
            head,
            "sudo -n k3s kubectl create namespace inference --dry-run=client -o yaml | "
            "sudo -n k3s kubectl apply -f - >/dev/null\n"
            "sudo -n k3s kubectl apply -f - <<'MANIFEST'\n"
            + json.dumps({"apiVersion": "v1", "kind": "List", "items": [manifest, pvc]})
            + "\nMANIFEST\n",
        )
    elif step == "dashboard":
        await service.transport.upload(head, chart_archive())
        # Verify the official Helm archive against its published SHA256 before installation.
        await run(
            head,
            f"""
helmdir=$(mktemp -d)
trap 'rm -rf "$helmdir"' EXIT
case "$(uname -m)" in aarch64|arm64) arch=arm64;; x86_64|amd64) arch=amd64;; *) exit 1;; esac
helm_url=https://get.helm.sh/helm-{HELM_VERSION}-linux-$arch.tar.gz
curl -fsSL --connect-timeout 10 --max-time 180 "$helm_url" -o "$helmdir/helm.tar.gz"
curl -fsSL --connect-timeout 10 --max-time 60 "$helm_url.sha256sum" -o "$helmdir/helm.sha256"
expected=$(cut -d ' ' -f 1 \"$helmdir/helm.sha256\")
echo "$expected  $helmdir/helm.tar.gz" | sha256sum -c - >/dev/null
tar xzf "$helmdir/helm.tar.gz" -C "$helmdir"
sudo -n install -m 0755 "$helmdir/linux-$arch/helm" /usr/local/bin/helm
""",
            timeout=300,
        )
        token64 = base64.b64encode(service.settings.model_admin_token.encode()).decode()
        secret = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": "spark-cluster-admin", "namespace": "inference"},
            "type": "Opaque",
            "data": {"token": token64},
        }
        digest = service.settings.setup_image_digest or ""
        if digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise SetupFailure("SETUP_IMAGE_DIGEST must be a lowercase sha256 digest")
        shared_values = {
            "image": {
                "repository": service.settings.setup_image_repository,
                "tag": service.settings.setup_image_tag,
                "digest": digest,
            },
            "controlPlane": {
                "enabled": True,
                "clusterName": name,
                "mode": "live",
                "service": {"type": "NodePort", "nodePort": 30080},
            },
            "models": {
                "adminTokenSecret": "spark-cluster-admin",
                "persistence": {"enabled": True, "existingClaim": "spark-cluster-models"},
            },
            "runtime": {
                "enabled": True,
                "deferStart": True,
                "hostNetwork": True,
                "openAIBaseUrl": f"http://{head.address}:30000/v1",
                "service": {"type": "NodePort", "nodePort": 30000},
                "tensorParallelSize": len(nodes),
                "nodeSelector": {"spark-cluster.io/managed": "true"},
            },
            "agent": {"nodeSelector": {"spark-cluster.io/managed": "true"}},
        }
        controller_values = json.loads(json.dumps(shared_values))
        controller_values["runtime"]["enabled"] = False
        runtime_values = json.loads(json.dumps(shared_values))
        runtime_values["controlPlane"]["enabled"] = False
        runtime_values["agent"]["enabled"] = False
        await run(
            head,
            "sudo -n k3s kubectl apply -f - <<'SECRET' >/dev/null\n"
            + json.dumps(secret)
            + "\nSECRET\n"
            "sudo -n tee /var/lib/spark-cluster/bootstrap/controller-values.json "
            "<<'CONTROLLER_VALUES' >/dev/null\n"
            + json.dumps(controller_values)
            + "\nCONTROLLER_VALUES\n"
            "sudo -n tee /var/lib/spark-cluster/bootstrap/runtime-values.json "
            "<<'RUNTIME_VALUES' >/dev/null\n" + json.dumps(runtime_values) + "\nRUNTIME_VALUES\n"
            "sudo -n env KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm upgrade --install "
            "spark-cluster-controller /var/lib/spark-cluster/bootstrap/spark-cluster "
            "--namespace inference "
            "--values /var/lib/spark-cluster/bootstrap/controller-values.json "
            "--wait --timeout 10m >/dev/null 2>&1\n"
            "sudo -n env KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm upgrade --install "
            "spark-cluster-runtime /var/lib/spark-cluster/bootstrap/spark-cluster "
            "--namespace inference "
            "--values /var/lib/spark-cluster/bootstrap/runtime-values.json "
            "--wait --timeout 10m >/dev/null 2>&1\n",
            timeout=660,
        )
    elif step == "verify":
        # Check all selected nodes + actual pod readiness and a RWX mount from each node.
        await run(
            head,
            "sudo -n k3s kubectl -n inference rollout status deployment/spark-cluster "
            "--timeout=180s >/dev/null\n"
            "sudo -n k3s kubectl -n inference rollout status daemonset/spark-cluster-agent "
            "--timeout=180s >/dev/null\n"
            "sudo -n k3s kubectl -n inference wait --for=jsonpath='{.status.phase}'=Bound "
            "pvc/spark-cluster-models --timeout=120s >/dev/null\n"
            f"curl -fsS --max-time 15 http://{head.address}:30080/healthz >/dev/null\n",
            timeout=400,
        )
        for node in nodes:
            await run(
                node,
                "sudo -n mkdir -p /var/lib/spark-cluster/storage-check\n"
                "if mountpoint -q /var/lib/spark-cluster/storage-check; then "
                "sudo -n umount /var/lib/spark-cluster/storage-check; fi\n"
                "trap 'sudo -n umount /var/lib/spark-cluster/storage-check "
                "2>/dev/null || true' EXIT\n"
                f"sudo -n mount -t nfs -o vers=4.2 {cluster_ip(head)}:"
                "/var/lib/spark-cluster/models /var/lib/spark-cluster/storage-check\n"
                "sudo -n -u '#10001' sh -c 'checkfile=$(mktemp "
                '/var/lib/spark-cluster/storage-check/.check.XXXXXX); rm "$checkfile"\'\n',
                timeout=45,
            )
            service.log(job, "Shared model storage read/write verified", node.address)
