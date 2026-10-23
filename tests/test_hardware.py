from spark_cluster.hardware import build_hardware_topology
from spark_cluster.setup_models import SetupRequest

PIN = "SHA256:" + "A" * 43


def request(count, topology):
    return SetupRequest(
        name="hardware-lab",
        topology=topology,
        nodes=[
            {
                "address": f"192.168.50.{10 + index}",
                "username": "nvidia",
                "fingerprint": PIN,
                "role": "head" if index == 0 else "worker",
            }
            for index in range(count)
        ],
    )


def interface(name, address, prefix=30, *, carrier=True, speed=200_000):
    return {
        "name": name,
        "speed_mbps": speed,
        "addresses": [address],
        "address_info": [{"local": address, "prefixlen": prefix}],
        "up": carrier,
        "carrier": carrier,
        "mtu": 9000,
        "mac_address": "02:00:00:00:00:01",
        "pci_address": "0000:01:00.0",
        "driver": "mlx5_core",
        "rdma": True,
        "rdma_device": name.replace("en", "mlx5_"),
    }


def facts_for(plan, interfaces, *, verified=True):
    for index, ports in enumerate(interfaces):
        for number, port in enumerate(ports):
            port["mac_address"] = f"02:00:00:00:{index:02x}:{number:02x}"
    if verified:
        for ports in interfaces:
            for port in ports:
                peers = [peer for group in interfaces for peer in group if peer is not port]
                port["peer_checks"] = [
                    {"address": peer["addresses"][0], "reachable": True} for peer in peers
                ]
                port["neighbors"] = [
                    {
                        "dst": peer["addresses"][0],
                        "lladdr": peer["mac_address"],
                        "state": ["REACHABLE"],
                    }
                    for peer in peers
                ]
    return {
        node.address: {"hostname": f"spark-{index + 1}", "interfaces": interfaces[index]}
        for index, node in enumerate(plan.nodes)
    }


def test_two_spark_direct_cable_requires_verified_peer_interfaces():
    plan = request(2, "direct")
    topology = build_hardware_topology(
        plan,
        facts_for(
            plan,
            [[interface("enp1", "10.10.1.1")], [interface("enp1", "10.10.1.2")]],
        ),
    )
    assert topology.verdict == "pass"
    assert topology.expected_cables == topology.detected_cables == 1
    assert topology.links[0].source_port == "enp1"
    assert topology.links[0].speed_gbps == 200


def test_missing_direct_cable_produces_specific_visual_instruction():
    plan = request(2, "direct")
    topology = build_hardware_topology(
        plan,
        facts_for(
            plan,
            [
                [interface("enp1", "10.10.1.1", carrier=False, speed=0)],
                [interface("enp1", "10.10.1.2", carrier=False, speed=0)],
            ],
        ),
    )
    assert topology.verdict == "error"
    assert topology.detected_cables == 0
    assert topology.links[0].status == "error"
    assert "spark-1 port enp1" in topology.instructions[1].detail
    assert "spark-2 port enp1" in topology.instructions[1].detail


def test_three_spark_ring_requires_all_three_pairwise_links():
    plan = request(3, "direct")
    complete = facts_for(
        plan,
        [
            [interface("en-a-b", "10.10.1.1"), interface("en-a-c", "10.10.2.1")],
            [interface("en-b-a", "10.10.1.2"), interface("en-b-c", "10.10.3.1")],
            [interface("en-c-a", "10.10.2.2"), interface("en-c-b", "10.10.3.2")],
        ],
    )
    topology = build_hardware_topology(plan, complete)
    assert topology.verdict == "pass"
    assert topology.detected_cables == 3
    complete[plan.nodes[2].address]["interfaces"][1]["carrier"] = False
    complete[plan.nodes[2].address]["interfaces"][1]["up"] = False
    complete[plan.nodes[2].address]["interfaces"][1]["speed_mbps"] = 0
    topology = build_hardware_topology(plan, complete)
    assert topology.verdict == "error"
    assert topology.detected_cables == 2
    assert any(link.status == "error" for link in topology.links)


def test_four_spark_switch_needs_one_healthy_link_per_node():
    plan = request(4, "switch")
    topology = build_hardware_topology(
        plan,
        facts_for(
            plan,
            [[interface(f"enp{index}", f"10.20.0.{10 + index}", prefix=24)] for index in range(4)],
        ),
    )
    assert topology.verdict == "pass"
    assert topology.expected_cables == topology.detected_cables == 4
    assert all(link.target == "switch" for link in topology.links)


def test_four_spark_ring_uses_two_unique_ports_per_node():
    plan = request(4, "ring")
    ports = [
        [
            interface(f"en-{index}-next", f"10.30.{index + 1}.1"),
            interface(f"en-{index}-prev", f"10.30.{index or 4}.2"),
        ]
        for index in range(4)
    ]
    # Each segment has its own /30: 0-1, 1-2, 2-3 and 3-0.
    ports[1][1]["addresses"] = ["10.30.1.2"]
    ports[1][1]["address_info"] = [{"local": "10.30.1.2", "prefixlen": 30}]
    ports[2][1]["addresses"] = ["10.30.2.2"]
    ports[2][1]["address_info"] = [{"local": "10.30.2.2", "prefixlen": 30}]
    ports[3][1]["addresses"] = ["10.30.3.2"]
    ports[3][1]["address_info"] = [{"local": "10.30.3.2", "prefixlen": 30}]
    ports[0][1]["addresses"] = ["10.30.4.2"]
    ports[0][1]["address_info"] = [{"local": "10.30.4.2", "prefixlen": 30}]
    topology = build_hardware_topology(plan, facts_for(plan, ports))
    assert topology.verdict == "pass"
    assert topology.expected_cables == topology.detected_cables == 4
    used = {}
    for link in topology.links:
        used.setdefault(link.source, set()).add(link.source_port)
        used.setdefault(link.target, set()).add(link.target_port)
    assert all(len(names) == 2 for names in used.values())
    assert any("NCCL ring" in item.title for item in topology.instructions)


def test_switch_carrier_without_common_network_is_not_declared_correct():
    plan = request(3, "switch")
    topology = build_hardware_topology(
        plan,
        facts_for(
            plan,
            [
                [interface("enp1", "10.20.1.10")],
                [interface("enp1", "10.20.2.10")],
                [interface("enp1", "10.20.3.10")],
            ],
        ),
    )
    assert topology.verdict == "warning"
    assert all(link.status == "warning" for link in topology.links)
    assert any("common switched subnet" in link.message for link in topology.links)


def test_slow_negotiated_link_is_a_warning_with_remedy():
    plan = request(2, "direct")
    topology = build_hardware_topology(
        plan,
        facts_for(
            plan,
            [
                [interface("enp1", "10.10.1.1", speed=100_000)],
                [interface("enp1", "10.10.1.2", speed=100_000)],
            ],
        ),
    )
    assert topology.verdict == "warning"
    assert topology.links[0].speed_gbps == 100
    assert any(item.status == "warning" for item in topology.instructions)


def test_single_spark_requires_no_fabric_cable():
    plan = request(1, "single")
    topology = build_hardware_topology(plan, facts_for(plan, [[]]))
    assert topology.verdict == "pass"
    assert topology.expected_cables == topology.detected_cables == 0
    assert "no cluster fabric cable" in topology.summary.lower()


def test_shared_subnet_alone_cannot_verify_cabling():
    plan = request(2, "direct")
    facts = facts_for(
        plan, [[interface("enp1", "10.10.1.1")], [interface("enp1", "10.10.1.2")]], verified=False
    )
    assert build_hardware_topology(plan, facts).verdict == "warning"


def test_wrong_peer_mac_and_failed_ping_cannot_verify_cabling():
    plan = request(2, "direct")
    for bad_mac in (True, False):
        facts = facts_for(
            plan, [[interface("enp1", "10.10.1.1")], [interface("enp1", "10.10.1.2")]]
        )
        port = facts[plan.nodes[0].address]["interfaces"][0]
        if bad_mac:
            port["neighbors"][0]["lladdr"] = "02:ff:ff:ff:ff:ff"
        else:
            port["peer_checks"][0]["reachable"] = False
        assert build_hardware_topology(plan, facts).verdict == "warning"


def test_three_sparks_with_one_shared_port_each_cannot_form_direct_triangle():
    plan = request(3, "direct")
    facts = facts_for(
        plan, [[interface("enp1", f"10.10.1.{index + 1}", prefix=24)] for index in range(3)]
    )
    topology = build_hardware_topology(plan, facts)
    assert topology.verdict != "pass"
    assert all(link.status != "pass" for link in topology.links)


def test_agent_does_not_report_total_accelerator_memory_as_kv_cache(monkeypatch):
    from spark_cluster.agent import collect_node

    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1_000_000, "SC_AVPHYS_PAGES": 500_000}
    monkeypatch.setattr("spark_cluster.agent.os.sysconf", values.__getitem__)
    monkeypatch.setattr("spark_cluster.agent.socket.gethostbyname", lambda _: "127.0.0.1")
    node = collect_node("test-node")
    assert node.memory.kv_cache_gib == 0
    assert node.memory.kv_cache_source == "unavailable"
