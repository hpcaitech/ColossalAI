"""Unit tests for cluster node isolation guardrails.

Verifies:
- node6 (192.168.0.98) is primary dev machine.
- node1 (192.168.0.46) is validation-only machine when idle.
- node5, node7, node8, node9 are strictly forbidden.
- node1 idle detection relies exclusively on device memory and process table,
  never on the flawed npu-smi AICore% counter.
"""

import socket
import pytest

from colossalai.cluster.node_guard import (
    NodeIsolationViolationError,
    assert_current_host_allowed,
    assert_nodes_allowed,
    check_node1_idle,
    is_forbidden_node,
    parse_npu_smi_proc_mem,
)


def test_forbidden_node_detection():
    forbidden = [
        "node5", "node7", "node8", "node9",
        "192.168.0.6", "192.168.0.138", "192.168.0.114", "192.168.0.88",
        "npu-910b2-6", "npu-910b2-138", "npu-910b2-114", "npu-910b2-88",
    ]
    for node in forbidden:
        assert is_forbidden_node(node) is True
        with pytest.raises(NodeIsolationViolationError, match="strictly forbidden"):
            assert_nodes_allowed(node)


def test_primary_node6_allowed_for_1node():
    for node in ["node6", "192.168.0.98", "npu-910b2-98"]:
        assert assert_nodes_allowed(node, allow_2node=False) is True
        assert assert_nodes_allowed([node], allow_2node=False) is True


def test_node1_rejected_for_1node_mode():
    for node in ["node1", "192.168.0.46", "npu-910b2-46"]:
        with pytest.raises(NodeIsolationViolationError, match="permitted only for 2-node validation"):
            assert_nodes_allowed(node, allow_2node=False)


def test_2node_mode_allows_node6_and_node1():
    valid_pairs = [
        ["node6", "node1"],
        "node6,node1",
        ["192.168.0.98", "192.168.0.46"],
        "192.168.0.98,192.168.0.46",
    ]
    for pair in valid_pairs:
        assert assert_nodes_allowed(pair, allow_2node=True) is True


def test_2node_mode_rejects_forbidden_nodes():
    invalid_pairs = [
        ["node6", "node5"],
        ["node6", "node7"],
        ["node6", "192.168.0.114"],
        "node6,node9",
    ]
    for pair in invalid_pairs:
        with pytest.raises(NodeIsolationViolationError, match="strictly forbidden"):
            assert_nodes_allowed(pair, allow_2node=True)


def test_2node_mode_requires_primary_node6():
    with pytest.raises(NodeIsolationViolationError, match="requires node6 as the primary"):
        assert_nodes_allowed(["node1", "node1"], allow_2node=True)


def test_proc_mem_parsing():
    idle_out = "npu-smi 23.0.0\nNo process in device.\n"
    res = parse_npu_smi_proc_mem(idle_out)
    assert res["idle"] is True
    assert res["active_processes"] == 0
    assert res["used_memory_bytes"] == 0

    busy_out = "Process id : 54321\nProcess name: python3\nMemory: 16384 MB\n"
    res_busy = parse_npu_smi_proc_mem(busy_out)
    assert res_busy["idle"] is False
    assert res_busy["active_processes"] >= 1
    assert res_busy["used_memory_bytes"] > 0


def test_node1_idle_check_success():
    def mock_idle_runner(cmd):
        if "proc-mem" in cmd:
            return "npu-smi 23.0.0\nNo process in device.\n", "", 0
        if "pgrep" in cmd:
            return "0\n", "", 0
        return "", "", 0

    is_idle, reason = check_node1_idle(runner=mock_idle_runner, num_devices=8)
    assert is_idle is True
    assert "verified idle" in reason


def test_node1_idle_check_fails_on_busy_device():
    def mock_busy_device_runner(cmd):
        if "-i 3" in cmd:
            return "Process id : 9988\nMemory: 8192 MB\n", "", 0
        if "proc-mem" in cmd:
            return "No process in device.\n", "", 0
        return "0\n", "", 0

    is_idle, reason = check_node1_idle(runner=mock_busy_device_runner, num_devices=8)
    assert is_idle is False
    assert "Device 3 on node1 is busy" in reason


def test_node1_idle_check_fails_on_running_pgrep():
    def mock_busy_pgrep_runner(cmd):
        if "proc-mem" in cmd:
            return "No process in device.\n", "", 0
        if "pgrep" in cmd:
            return "2\n", "", 0
        return "", "", 0

    is_idle, reason = check_node1_idle(runner=mock_busy_pgrep_runner, num_devices=8)
    assert is_idle is False
    assert "training/precomputation process" in reason


def test_assert_current_host_allowed(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "npu-910b2-98")
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda host: ("npu-910b2-98", [], ["192.168.0.98"]))
    assert_current_host_allowed(allow_secondary=False)

    # Node 1 without allow_secondary should fail
    monkeypatch.setattr(socket, "gethostname", lambda: "npu-910b2-46")
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda host: ("npu-910b2-46", [], ["192.168.0.46"]))
    with pytest.raises(NodeIsolationViolationError, match="is node1, but allow_secondary is False"):
        assert_current_host_allowed(allow_secondary=False)

    # Node 1 with allow_secondary should pass
    assert_current_host_allowed(allow_secondary=True)

    # Forbidden node 5 should fail unconditionally
    monkeypatch.setattr(socket, "gethostname", lambda: "npu-910b2-6")
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda host: ("npu-910b2-6", [], ["192.168.0.6"]))
    with pytest.raises(NodeIsolationViolationError, match="FORBIDDEN node"):
        assert_current_host_allowed(allow_secondary=True)


def test_compound_string_rejection():
    # Compound strings with forbidden hostnames
    with pytest.raises(NodeIsolationViolationError, match="strictly forbidden"):
        assert_nodes_allowed("node6-npu-910b2-6", allow_2node=False)

    # Compound strings with unauthorized external hostnames
    with pytest.raises(NodeIsolationViolationError, match="not authorized"):
        assert_nodes_allowed("node6-worker-aws", allow_2node=False)

    # Compound strings in 2-node mode list
    with pytest.raises(NodeIsolationViolationError, match="strictly forbidden"):
        assert_nodes_allowed(["node6", "node1-npu-910b2-6"], allow_2node=True)


def test_fqdn_host_isolation(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "npu-910b2-6.local")
    monkeypatch.setattr(socket, "gethostbyname_ex", lambda host: ("npu-910b2-6.local", [], ["10.0.0.5"]))
    with pytest.raises(NodeIsolationViolationError, match="FORBIDDEN node"):
        assert_current_host_allowed(allow_secondary=False)

    with pytest.raises(NodeIsolationViolationError, match="FORBIDDEN node"):
        assert_current_host_allowed(allow_secondary=True)


def test_check_node1_idle_fails_on_command_error():
    # Non-zero exit code
    failing_runner = lambda cmd: ("", "Error: npu-smi crashed", 1)
    is_idle, reason = check_node1_idle(runner=failing_runner)
    assert is_idle is False
    assert "Failed to query device" in reason

    # Empty stdout
    empty_runner = lambda cmd: ("", "", 0)
    is_idle, reason = check_node1_idle(runner=empty_runner)
    assert is_idle is False
    assert "Failed to query device" in reason


def test_2node_mode_with_single_node1_rejected():
    with pytest.raises(NodeIsolationViolationError, match="requires node6 as the primary"):
        assert_nodes_allowed(["node1"], allow_2node=True)

    with pytest.raises(NodeIsolationViolationError, match="requires node6 as the primary"):
        assert_nodes_allowed("node1", allow_2node=True)

