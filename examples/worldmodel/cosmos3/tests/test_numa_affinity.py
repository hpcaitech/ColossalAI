"""CPU contracts for the NUMA pin map: parsing, rank selection, and the sysfs-derived socket cpusets."""

import os

import pytest

from examples.worldmodel.cosmos3 import numa_affinity as numa

NPU_SMI = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 25.2.1                   Version: 25.2.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 341.0       58                0    / 0             |
| 0                         | 0000:C1:00.0  | 54          0    / 0          61503/ 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 394.0       60                0    / 0             |
| 0                         | 0000:C2:00.0  | 75          0    / 0          57457/ 65536         |
+===========================+===============+====================================================+
"""


def make_sysfs(root, domains):
    """domains: list of (cpulist, package) in nodeN order."""
    for index, (cpulist, package) in enumerate(domains):
        node = root / f"devices/system/node/node{index}"
        node.mkdir(parents=True)
        (node / "cpulist").write_text(cpulist + "\n")
        for cpu in numa.parse_cpulist(cpulist):
            cpu_dir = root / f"devices/system/cpu/cpu{cpu}/topology"
            cpu_dir.mkdir(parents=True, exist_ok=True)
            (cpu_dir / "physical_package_id").write_text(f"{package}\n")
    (root / "bus/pci/devices").mkdir(parents=True)


def add_device(root, bus, domain):
    dev = root / f"bus/pci/devices/{bus.lower()}"
    dev.mkdir(parents=True, exist_ok=True)
    (dev / "numa_node").write_text(f"{domain}\n")


def test_cpulist_round_trip():
    for text in ("0-23", "0-2,5", "144-191", "0,2,4,6"):
        assert numa.format_cpulist(numa.parse_cpulist(text)) == text
    assert numa.parse_cpulist("0-1,3") == [0, 1, 3]
    assert numa.format_cpulist([]) == ""


def test_spec_and_rank_selection():
    spec = "144-191;144-191;96-143"
    assert numa.parse_spec(spec) == ["144-191", "144-191", "96-143"]
    assert numa.affinity_for_rank(spec, 0) == "144-191"
    assert numa.affinity_for_rank(spec, 2) == "96-143"
    # More ranks than entries: wrap, so a 4-domain node still pins every rank.
    assert numa.affinity_for_rank(spec, 3) == "144-191"
    for bad in ("", "  ", "0-1;;2-3", "0-x"):
        with pytest.raises(ValueError):
            numa.parse_spec(bad)


def test_device_buses_follow_npu_smi_order():
    assert numa.parse_device_buses(NPU_SMI) == ["0000:C1:00.0", "0000:C2:00.0"]


def test_socket_map_nine_domains(tmp_path):
    # node1/6/7/8: 8 domains of 24 cores, two domains per socket, devices on the even domains.
    make_sysfs(tmp_path, [(f"{24 * i}-{24 * i + 23}", i // 2) for i in range(8)])
    buses = ["0000:C1:00.0", "0000:C2:00.0", "0000:81:00.0", "0000:82:00.0"]
    for bus, domain in zip(buses, (6, 6, 4, 4)):
        add_device(tmp_path, bus, domain)
    assert numa.build_map(buses, tmp_path) == ["144-191", "144-191", "96-143", "96-143"]


def test_socket_map_four_domains(tmp_path):
    # node5/node9: 4 domains of 48 cores, one per socket.
    make_sysfs(tmp_path, [(f"{48 * i}-{48 * i + 47}", i) for i in range(4)])
    buses = ["0000:C1:00.0", "0000:01:00.0"]
    add_device(tmp_path, buses[0], 3)
    add_device(tmp_path, buses[1], 0)
    assert numa.build_map(buses, tmp_path) == ["144-191", "0-47"]


def test_missing_numa_node_is_an_error(tmp_path):
    make_sysfs(tmp_path, [("0-47", 0)])
    add_device(tmp_path, "0000:01:00.0", -1)
    with pytest.raises(ValueError):
        numa.build_map(["0000:01:00.0"], tmp_path)


def test_derive_map_runs_npu_smi_and_restricts_to_visible_devices(tmp_path, monkeypatch):
    make_sysfs(tmp_path, [(f"{48 * i}-{48 * i + 47}", i) for i in range(4)])
    add_device(tmp_path, "0000:C1:00.0", 3)
    add_device(tmp_path, "0000:C2:00.0", 0)
    npu_smi = tmp_path / "npu-smi"
    npu_smi.write_text("#!/bin/sh\ncat <<'X'\n" + NPU_SMI + "X\n")
    npu_smi.chmod(0o755)

    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert numa.derive_map(str(npu_smi), tmp_path) == "144-191;0-47"
    # A subset of cards must index the map by the visible order, not the physical device index.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    assert numa.derive_map(str(npu_smi), tmp_path) == "0-47"


linux_only = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="affinity syscalls are Linux-only")


@linux_only
def test_apply_affinity_is_a_noop_without_the_variable(monkeypatch):
    monkeypatch.delenv("COSMOS3_NUMA_PIN", raising=False)
    before = os.sched_getaffinity(0)
    applied, note = numa.apply_affinity()
    assert applied is False and "unset" in note
    assert os.sched_getaffinity(0) == before


@linux_only
def test_apply_affinity_pins_the_current_process(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    target = allowed[: max(1, len(allowed) // 2)]
    spec = numa.format_cpulist(target)
    monkeypatch.setenv("LOCAL_RANK", "0")
    try:
        applied, note = numa.apply_affinity(spec=spec)
        assert applied is True and spec in note
        assert os.sched_getaffinity(0) == set(target)
    finally:
        os.sched_setaffinity(0, allowed)


@linux_only
def test_apply_affinity_auto_derives_this_nodes_map(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    target = allowed[: max(1, len(allowed) // 2)]
    spec = numa.format_cpulist(target)
    monkeypatch.setenv("COSMOS3_NUMA_PIN", "auto")
    monkeypatch.setattr(numa, "derive_map", lambda *a, **k: spec)
    try:
        applied, note = numa.apply_affinity()
        assert applied is True and spec in note
        assert os.sched_getaffinity(0) == set(target)
    finally:
        os.sched_setaffinity(0, allowed)


@linux_only
def test_apply_affinity_auto_failure_is_a_noop(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("npu-smi missing")

    monkeypatch.setenv("COSMOS3_NUMA_PIN", "auto")
    monkeypatch.setattr(numa, "derive_map", boom)
    before = os.sched_getaffinity(0)
    applied, note = numa.apply_affinity()
    assert applied is False and "auto" in note and "npu-smi missing" in note
    assert os.sched_getaffinity(0) == before
