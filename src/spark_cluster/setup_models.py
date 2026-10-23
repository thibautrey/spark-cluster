"""Public setup contracts. No private keys, passwords, or enrollment tokens."""

import ipaddress
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def private_address(value: str) -> str:
    address = ipaddress.IPv4Address(value)
    allowed = any(
        address in ipaddress.ip_network(cidr)
        for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    )
    if not allowed:
        raise ValueError("Use an RFC1918 LAN IPv4 address")
    return str(address)


class DiscoveryRequest(BaseModel):
    subnet: str | None = None
    addresses: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("subnet")
    @classmethod
    def bounded_subnet(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        network = ipaddress.IPv4Network(value, strict=False)
        if network.prefixlen < 24:
            raise ValueError("Scan at most one /24 network (254 hosts)")
        private_address(str(network.network_address))
        private_address(str(network.broadcast_address))
        return str(network)

    @field_validator("addresses")
    @classmethod
    def lan_addresses(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(private_address(value) for value in values))


class SetupNode(BaseModel):
    address: str
    username: str = Field(default="nvidia", pattern=r"^[a-z_][a-z0-9_-]{0,31}$")
    fingerprint: str = Field(pattern=r"^SHA256:[A-Za-z0-9+/]{43}$")
    role: Literal["head", "worker"] = "worker"
    fabric_address: str | None = None

    _address = field_validator("address")(private_address)

    @field_validator("fabric_address")
    @classmethod
    def fabric_lan(cls, value: str | None) -> str | None:
        return private_address(value) if value else None


class SetupRequest(BaseModel):
    name: str = Field(default="spark-cluster", pattern=r"^[a-z][a-z0-9-]{0,39}$")
    nodes: list[SetupNode] = Field(min_length=1, max_length=4)
    topology: Literal["single", "direct", "switch"] = "switch"

    @model_validator(mode="after")
    def valid_cluster(self):
        if len({node.address for node in self.nodes}) != len(self.nodes):
            raise ValueError("Each machine may only be selected once")
        if sum(node.role == "head" for node in self.nodes) != 1:
            raise ValueError("Select exactly one head node")
        if len(self.nodes) == 4 and self.topology != "switch":
            raise ValueError("Four Sparks require a ConnectX-7 switch")
        if (len(self.nodes) == 1) != (self.topology == "single"):
            raise ValueError("Use single topology for one node only")
        return self


class ClusterConnectRequest(BaseModel):
    address: str
    port: int = Field(default=30080, ge=1024, le=65535)

    _address = field_validator("address")(private_address)


class SetupCheck(BaseModel):
    node: str
    code: str
    status: Literal["pass", "warning", "error"]
    message: str
    remedy: str = ""
    automatic: bool = False


class FabricPort(BaseModel):
    name: str
    status: Literal["connected", "slow", "disconnected", "unknown"]
    carrier: bool
    speed_gbps: int = 0
    rdma: bool = False
    rdma_device: str | None = None
    pci_address: str | None = None
    mac_address: str | None = None
    mtu: int = 0
    addresses: list[str] = Field(default_factory=list)


class HardwareNode(BaseModel):
    address: str
    hostname: str
    role: Literal["head", "worker"]
    status: Literal["pass", "warning", "error", "pending"]
    ports: list[FabricPort] = Field(default_factory=list)


class FabricLink(BaseModel):
    source: str
    target: str
    status: Literal["pass", "warning", "error", "pending"]
    source_port: str | None = None
    target_port: str | None = None
    speed_gbps: int = 0
    message: str


class HardwareInstruction(BaseModel):
    status: Literal["pass", "warning", "error", "info"]
    title: str
    detail: str


class HardwareTopology(BaseModel):
    topology: Literal["single", "direct", "switch"]
    verdict: Literal["pass", "warning", "error", "pending"]
    expected_cables: int
    detected_cables: int
    summary: str
    nodes: list[HardwareNode]
    links: list[FabricLink]
    instructions: list[HardwareInstruction]


class SetupStep(BaseModel):
    id: str
    label: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    message: str = ""


class SetupJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    request: SetupRequest
    mode: str
    status: Literal["queued", "running", "blocked", "failed", "ready"] = "queued"
    steps: list[SetupStep]
    checks: list[SetupCheck] = Field(default_factory=list)
    hardware_topology: HardwareTopology | None = None
    logs: list[dict[str, str]] = Field(default_factory=list)
    dashboard_url: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
