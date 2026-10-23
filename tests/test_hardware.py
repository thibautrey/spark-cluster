from spark_deepseek.hardware import build_hardware_topology
from spark_deepseek.setup_models import SetupRequest

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


def facts_for(plan, interfaces):
    return {
        node.address: {"hostname": f"spark-{index + 1}", "interfaces": interfaces[index]}
        for index, node in enumerate(plan.nodes)
    }


def test_two_spark_direct_cable_is_detected_from_shared_subnet():
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
            [
                [interface(f"enp{index}", f"10.20.0.{10 + index}", prefix=24)]
                for index in range(4)
            ],
        ),
    )
    assert topology.verdict == "pass"
    assert topology.expected_cables == topology.detected_cables == 4
    assert all(link.target == "switch" for link in topology.links)


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
