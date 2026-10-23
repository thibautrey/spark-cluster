"""Turn read-only NIC/RDMA facts into an actionable physical cabling plan."""

import ipaddress
import itertools
from typing import Any

from .setup_models import (
    FabricLink,
    FabricPort,
    HardwareInstruction,
    HardwareNode,
    HardwareTopology,
    SetupRequest,
)

MIN_LINK_GBPS = 184


def _port(interface: dict[str, Any]) -> FabricPort:
    speed = max(0, int(interface.get("speed_mbps", 0) or 0) // 1000)
    carrier = bool(interface.get("carrier", interface.get("up", False)))
    if carrier and speed >= MIN_LINK_GBPS:
        status = "connected"
    elif carrier:
        status = "slow"
    else:
        status = "disconnected"
    return FabricPort(
        name=str(interface.get("name", "unknown")),
        status=status,
        carrier=carrier,
        speed_gbps=speed,
        rdma=bool(interface.get("rdma", False)),
        rdma_device=interface.get("rdma_device"),
        pci_address=interface.get("pci_address"),
        mac_address=interface.get("mac_address"),
        mtu=int(interface.get("mtu", 0) or 0),
        addresses=list(interface.get("addresses", [])),
    )


def _fabric_ports(facts: dict[str, Any]) -> list[FabricPort]:
    interfaces = facts.get("interfaces", [])
    selected = [
        interface
        for interface in interfaces
        if interface.get("rdma")
        or int(interface.get("speed_mbps", 0) or 0) >= 100_000
        or str(interface.get("driver", "")).startswith("mlx5")
    ]
    return [_port(interface) for interface in selected]


def _networks(port: FabricPort, raw: dict[str, Any]) -> set[ipaddress.IPv4Network]:
    interface = next(
        (item for item in raw.get("interfaces", []) if item.get("name") == port.name), {}
    )
    values = set()
    for address in interface.get("address_info", []):
        try:
            values.add(
                ipaddress.ip_network(f"{address['local']}/{address['prefixlen']}", strict=False)
            )
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _node_status(ports: list[FabricPort], single: bool) -> str:
    if single:
        return "pass"
    if not ports or not any(port.carrier for port in ports):
        return "error"
    if any(port.status == "slow" for port in ports):
        return "warning"
    return "pass"


def _suggested_port(node: HardwareNode) -> str:
    available = next((port.name for port in node.ports if not port.carrier), None)
    return available or (node.ports[0].name if node.ports else "ConnectX-7 QSFP port")


def _direct_link(
    left: HardwareNode,
    right: HardwareNode,
    left_facts: dict[str, Any],
    right_facts: dict[str, Any],
    node_count: int,
) -> FabricLink:
    for left_port in left.ports:
        if not left_port.carrier:
            continue
        for right_port in right.ports:
            if not right_port.carrier:
                continue
            common = _networks(left_port, left_facts) & _networks(right_port, right_facts)
            if common:
                speed = min(left_port.speed_gbps, right_port.speed_gbps)
                status = "pass" if speed >= MIN_LINK_GBPS else "warning"
                return FabricLink(
                    source=left.address,
                    target=right.address,
                    source_port=left_port.name,
                    target_port=right_port.name,
                    speed_gbps=speed,
                    status=status,
                    message=(
                        f"Direct cable detected at {speed} Gbit/s"
                        if status == "pass"
                        else f"Cable detected but negotiated only {speed} Gbit/s"
                    ),
                )
    if node_count == 2:
        left_up = next((port for port in left.ports if port.carrier), None)
        right_up = next((port for port in right.ports if port.carrier), None)
        if left_up and right_up:
            speed = min(left_up.speed_gbps, right_up.speed_gbps)
            return FabricLink(
                source=left.address,
                target=right.address,
                source_port=left_up.name,
                target_port=right_up.name,
                speed_gbps=speed,
                status="warning",
                message="Both ports have carrier, but their point-to-point subnet is not aligned",
            )
    return FabricLink(
        source=left.address,
        target=right.address,
        source_port=_suggested_port(left),
        target_port=_suggested_port(right),
        status="error",
        message="Expected direct cable is not detected",
    )


def build_hardware_topology(
    request: SetupRequest, facts: dict[str, dict[str, Any]], *, simulated: bool = False
) -> HardwareTopology:
    nodes = []
    for requested in request.nodes:
        raw = facts.get(requested.address, {})
        ports = _fabric_ports(raw)
        if simulated and request.topology != "single":
            count = 2 if request.topology == "direct" and len(request.nodes) == 3 else 1
            ports = [
                FabricPort(
                    name=f"cx7-{index + 1}",
                    status="connected",
                    carrier=True,
                    speed_gbps=200,
                    rdma=True,
                    rdma_device=f"mlx5_{index}",
                    mtu=9000,
                    addresses=[f"10.10.{index}.{10 + len(nodes)}"],
                )
                for index in range(count)
            ]
        nodes.append(
            HardwareNode(
                address=requested.address,
                hostname=str(raw.get("hostname") or requested.address),
                role=requested.role,
                status="pass" if simulated else _node_status(ports, request.topology == "single"),
                ports=ports,
            )
        )

    links: list[FabricLink] = []
    if request.topology == "direct":
        simulated_ports = {node.address: 0 for node in nodes}
        for left_index, right_index in itertools.combinations(range(len(nodes)), 2):
            left, right = nodes[left_index], nodes[right_index]
            if simulated:
                left_port = left.ports[simulated_ports[left.address] % len(left.ports)]
                right_port = right.ports[simulated_ports[right.address] % len(right.ports)]
                simulated_ports[left.address] += 1
                simulated_ports[right.address] += 1
                link = FabricLink(
                    source=left.address,
                    target=right.address,
                    source_port=left_port.name,
                    target_port=right_port.name,
                    speed_gbps=200,
                    status="pass",
                    message="Simulated direct cable at 200 Gbit/s",
                )
            else:
                link = _direct_link(
                    left,
                    right,
                    facts.get(left.address, {}),
                    facts.get(right.address, {}),
                    len(nodes),
                )
            links.append(link)
    elif request.topology == "switch":
        for node in nodes:
            connected = next((port for port in node.ports if port.carrier), None)
            status = (
                "error"
                if connected is None
                else ("pass" if connected.status == "connected" else "warning")
            )
            links.append(
                FabricLink(
                    source=node.address,
                    target="switch",
                    source_port=connected.name if connected else _suggested_port(node),
                    speed_gbps=connected.speed_gbps if connected else 0,
                    status="pass" if simulated else status,
                    message=(
                        "Cable to switch detected at 200 Gbit/s"
                        if simulated
                        else "No cable to the switch is detected"
                        if connected is None
                        else f"Switch link detected at {connected.speed_gbps} Gbit/s"
                    ),
                )
            )
        if not simulated and nodes:
            node_networks = []
            for node in nodes:
                networks = set()
                for port in node.ports:
                    if port.carrier:
                        networks |= _networks(port, facts.get(node.address, {}))
                node_networks.append(networks)
            common_networks = set.intersection(*node_networks) if all(node_networks) else set()
            if not common_networks:
                for link in links:
                    if link.status == "pass":
                        link.status = "warning"
                        link.message = (
                            "Port carrier is present, but a common switched subnet is not verified"
                        )

    expected = (
        0
        if request.topology == "single"
        else (len(request.nodes) if request.topology == "switch" else len(links))
    )
    detected = sum(link.status in {"pass", "warning"} for link in links)
    errors = [link for link in links if link.status == "error"]
    warnings = [link for link in links if link.status == "warning"]
    instructions = [
        HardwareInstruction(
            status="info",
            title="Use one supported cable per link",
            detail=(
                "Use a QSFP112 DAC, 400 GbE, Ethernet-mode cable. Supported examples include "
                "Amphenol NJAAKK-N911 and Luxshare LMTQF022-SD-R. Fully seat both ends."
            ),
        )
    ]
    labels = {node.address: node.hostname for node in nodes}
    for link in errors:
        source = labels.get(link.source, link.source)
        target = "the switch" if link.target == "switch" else labels.get(link.target, link.target)
        detail = f"Connect {source} port {link.source_port} to {target}"
        if link.target_port:
            detail += f" port {link.target_port}"
        detail += ", then wait for the port LEDs and run detection again."
        instructions.append(
            HardwareInstruction(status="error", title="Connect the missing cable", detail=detail)
        )
    for link in warnings:
        instructions.append(
            HardwareInstruction(
                status="warning",
                title="Link needs attention",
                detail=(
                    f"{link.message}. Reseat the cable and verify Ethernet mode and switch speed."
                ),
            )
        )
    if request.topology == "single":
        verdict = "pass"
        summary = "One Spark selected — no cluster fabric cable is required."
        instructions = [
            HardwareInstruction(
                status="pass",
                title="No fabric cabling needed",
                detail="Connect management Ethernet and power. You can add fabric cabling later.",
            )
        ]
    elif errors:
        verdict = "error"
        summary = f"{detected} of {expected} expected cables detected. Follow the diagram."
    elif warnings:
        verdict = "warning"
        summary = f"All {expected} cables have carrier, but one or more links need attention."
    else:
        verdict = "pass"
        summary = f"All {expected} expected cables are detected at the required speed."
        instructions.append(
            HardwareInstruction(
                status="pass",
                title="Physical topology looks correct",
                detail=(
                    "Continue with NVIDIA Sync network configuration and the full readiness test."
                ),
            )
        )
    return HardwareTopology(
        topology=request.topology,
        verdict=verdict,
        expected_cables=expected,
        detected_cables=detected,
        summary=summary,
        nodes=nodes,
        links=links,
        instructions=instructions,
    )
