"""Fixed bootstrap operations. Inputs are validated contracts and shell-quoted."""

import base64
import io
import json
import shlex
import tarfile
from pathlib import Path

from .cluster_setup import SetupFailure

K3S_VERSION = "v1.34.1+k3s1"
HELM_VERSION = "v3.19.0"
GPU_PLUGIN_VERSION = "v0.17.4"


def chart_archive() -> bytes:
    chart = Path(__file__).parent / "chart"
    if not (chart / "Chart.yaml").exists():
        raise SetupFailure("The packaged Helm chart is unavailable")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        archive.add(chart, arcname="spark-deepseek")
    return output.getvalue()


def cluster_ip(node):
    return node.fabric_address or node.address


def guard(name, role):
    return (
        "if command -v k3s >/dev/null; then\n"
        f'  test "$(cat /var/lib/spark-deepseek/cluster-name 2>/dev/null)" = {shlex.quote(name)}\n'
        f'  test "$(cat /var/lib/spark-deepseek/cluster-role 2>/dev/null)" = {shlex.quote(role)}\n'
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
sudo -n mkdir -p /var/lib/spark-deepseek
"""
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
                "/var/lib/spark-deepseek/cluster-name >/dev/null\n"
                "echo head | sudo -n tee /var/lib/spark-deepseek/cluster-role >/dev/null\n"
            )
            + install_k3s(
                f"server --node-ip={ip} --advertise-address={ip} --tls-san={head.address} "
                "--disable=traefik --disable=servicelb --default-runtime=nvidia "
                "--write-kubeconfig-mode=600"
            )
            + "sudo -n k3s kubectl wait --for=condition=Ready nodes "
            "--all --timeout=240s >/dev/null\n"
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
                    "/var/lib/spark-deepseek/cluster-name >/dev/null\n"
                    "echo worker | sudo -n tee /var/lib/spark-deepseek/cluster-role >/dev/null\n"
                    "sudo -n install -m 600 /dev/null /var/lib/spark-deepseek/join.env\n"
                    f"printf '%s\\n' "
                    f"{shlex.quote('K3S_URL=https://' + cluster_ip(head) + ':6443')} "
                    f"{shlex.quote('K3S_TOKEN=' + shlex.quote(token))} | "
                    "sudo -n tee /var/lib/spark-deepseek/join.env >/dev/null\n"
                    "installer=$(mktemp)\n"
                    "trap 'sudo -n rm -f /var/lib/spark-deepseek/join.env; "
                    'rm -f "$installer"\' EXIT\n'
                    "curl -fsSL --connect-timeout 10 --max-time 120 https://get.k3s.io "
                    '-o "$installer"\n'
                )
                # Quote the complete root shell body once to preserve INSTALL_K3S_EXEC spaces.
                root = (
                    "set -a; . /var/lib/spark-deepseek/join.env; "
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
                    "spark-deepseek.io/managed=true --overwrite >/dev/null\n",
                )
        if actual != expected:
            raise SetupFailure("Not all selected nodes joined with an allocatable GPU")
    elif step == "storage":
        clients = " ".join(
            f"{cluster_ip(node)}(rw,sync,no_subtree_check,root_squash)" for node in nodes
        )
        export = f"/var/lib/spark-deepseek/models {clients}\n"
        encoded = base64.b64encode(export.encode()).decode()
        await run(
            head,
            """
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nfs-kernel-server >/dev/null 2>&1
sudo -n install -d -o 10001 -g 10001 -m 0770 /var/lib/spark-deepseek/models
"""
            + f"echo {shlex.quote(encoded)} | base64 -d | sudo -n tee "
            "/etc/exports.d/spark-deepseek.exports >/dev/null\n"
            "sudo -n exportfs -ra\nsudo -n systemctl enable --now nfs-server >/dev/null\n",
            timeout=600,
        )
        manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolume",
            "metadata": {
                "name": "spark-deepseek-models",
                "labels": {"spark-deepseek.io/storage": name},
            },
            "spec": {
                "capacity": {"storage": "1Ti"},
                "accessModes": ["ReadWriteMany"],
                "persistentVolumeReclaimPolicy": "Retain",
                "storageClassName": "",
                "mountOptions": ["nfsvers=4.2"],
                "nfs": {"server": cluster_ip(head), "path": "/var/lib/spark-deepseek/models"},
            },
        }
        pvc = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": "spark-deepseek-models", "namespace": "inference"},
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "storageClassName": "",
                "volumeName": "spark-deepseek-models",
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
case "$(uname -m)" in aarch64) arch=arm64;; *) exit 1;; esac
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
            "metadata": {"name": "spark-deepseek-admin", "namespace": "inference"},
            "type": "Opaque",
            "data": {"token": token64},
        }
        values = {
            "image": {
                "repository": service.settings.setup_image_repository,
                "tag": service.settings.setup_image_tag,
            },
            "controlPlane": {"mode": "live", "service": {"type": "NodePort", "nodePort": 30080}},
            "models": {
                "adminTokenSecret": "spark-deepseek-admin",
                "persistence": {"enabled": True, "existingClaim": "spark-deepseek-models"},
            },
            "runtime": {
                "enabled": True,
                "deferStart": True,
                "hostNetwork": True,
                "tensorParallelSize": len(nodes),
                "nodeSelector": {"spark-deepseek.io/managed": "true"},
            },
            "agent": {"nodeSelector": {"spark-deepseek.io/managed": "true"}},
        }
        await run(
            head,
            "sudo -n k3s kubectl apply -f - <<'SECRET' >/dev/null\n"
            + json.dumps(secret)
            + "\nSECRET\n"
            "sudo -n tee /var/lib/spark-deepseek/bootstrap/values.json "
            "<<'VALUES' >/dev/null\n" + json.dumps(values) + "\nVALUES\n"
            "sudo -n env KUBECONFIG=/etc/rancher/k3s/k3s.yaml helm upgrade --install "
            "spark-deepseek /var/lib/spark-deepseek/bootstrap/spark-deepseek --namespace inference "
            "--values /var/lib/spark-deepseek/bootstrap/values.json "
            "--wait --timeout 10m >/dev/null 2>&1\n",
            timeout=660,
        )
    elif step == "verify":
        # Check all selected nodes + actual pod readiness and a RWX mount from each node.
        await run(
            head,
            "sudo -n k3s kubectl -n inference rollout status deployment/spark-deepseek "
            "--timeout=180s >/dev/null\n"
            "sudo -n k3s kubectl -n inference rollout status daemonset/spark-deepseek-agent "
            "--timeout=180s >/dev/null\n"
            "sudo -n k3s kubectl -n inference wait --for=jsonpath='{.status.phase}'=Bound "
            "pvc/spark-deepseek-models --timeout=120s >/dev/null\n"
            f"curl -fsS --max-time 15 http://{head.address}:30080/healthz >/dev/null\n",
            timeout=400,
        )
        for node in nodes:
            await run(
                node,
                "sudo -n mkdir -p /var/lib/spark-deepseek/storage-check\n"
                "if mountpoint -q /var/lib/spark-deepseek/storage-check; then "
                "sudo -n umount /var/lib/spark-deepseek/storage-check; fi\n"
                "trap 'sudo -n umount /var/lib/spark-deepseek/storage-check "
                "2>/dev/null || true' EXIT\n"
                f"sudo -n mount -t nfs -o vers=4.2 {cluster_ip(head)}:"
                "/var/lib/spark-deepseek/models /var/lib/spark-deepseek/storage-check\n"
                "sudo -n -u '#10001' sh -c 'checkfile=$(mktemp "
                '/var/lib/spark-deepseek/storage-check/.check.XXXXXX); rm "$checkfile"\'\n',
                timeout=45,
            )
            service.log(job, "Shared model storage read/write verified", node.address)
