"""Cluster Node Isolation and Allocation Guardrail.

Enforces strict cluster topology constraints for Ascend 910B2 online training:
- node6 (192.168.0.98): Primary development, profiling, and iterative optimization machine.
- node1 (192.168.0.46): Exclusively for idle 2-node verification.
- node5, node7, node8, node9: Strictly untouched (zero SSH, zero jobs, zero data writes).

Idleness of node1 MUST be judged by inspecting the device process memory table
(npu-smi info -t proc-mem -i <dev>) and host/container process table (pgrep),
never by npu-smi AICore% (which is broken on Driver 25.2.1 and reads 89-99% even with zombie processes).
"""

import os
import re
import socket
import subprocess
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

NODE_TOPOLOGY: Dict[str, Dict[str, str]] = {
    "node6": {
        "hostname": "npu-910b2-98",
        "ip": "192.168.0.98",
        "role": "primary",
        "nvme": "/mnt/disk0",
    },
    "node1": {
        "hostname": "npu-910b2-46",
        "ip": "192.168.0.46",
        "role": "secondary_validation",
        "nvme": "/mnt/disk0",
    },
    "node5": {
        "hostname": "npu-910b2-6",
        "ip": "192.168.0.6",
        "role": "forbidden",
        "nvme": "",
    },
    "node7": {
        "hostname": "npu-910b2-138",
        "ip": "192.168.0.138",
        "role": "forbidden",
        "nvme": "",
    },
    "node8": {
        "hostname": "npu-910b2-114",
        "ip": "192.168.0.114",
        "role": "forbidden",
        "nvme": "",
    },
    "node9": {
        "hostname": "npu-910b2-88",
        "ip": "192.168.0.88",
        "role": "forbidden",
        "nvme": "",
    },
}

PRIMARY_IDENTIFIERS: Set[str] = {
    "node6",
    "npu-910b2-98",
    "192.168.0.98",
    "localhost",
    "127.0.0.1",
}

SECONDARY_IDENTIFIERS: Set[str] = {
    "node1",
    "npu-910b2-46",
    "192.168.0.46",
}

FORBIDDEN_IDENTIFIERS: Set[str] = {
    "node5", "npu-910b2-6", "192.168.0.6",
    "node7", "npu-910b2-138", "192.168.0.138",
    "node8", "npu-910b2-114", "192.168.0.114",
    "node9", "npu-910b2-88", "192.168.0.88",
}

FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
    "node5", "npu-910b2-6", "192.168.0.6",
    "node7", "npu-910b2-138", "192.168.0.138",
    "node8", "npu-910b2-114", "192.168.0.114",
    "node9", "npu-910b2-88", "192.168.0.88",
)


class NodeIsolationViolationError(PermissionError):
    """Raised when an operation targets a forbidden cluster node or violates isolation boundaries."""
    pass


def normalize_node_name(ident: str) -> str:
    """Normalize node identifier by stripping whitespace, scheme, path, port, and domain suffix."""
    clean = str(ident).strip().lower()
    if "://" in clean:
        clean = clean.split("://", 1)[1]
    if "/" in clean:
        clean = clean.split("/", 1)[0]
    if ":" in clean:
        clean = clean.split(":", 1)[0]
    # If not a standard IPv4 address, strip domain suffix (e.g. npu-910b2-6.local -> npu-910b2-6)
    if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", clean):
        clean = clean.split(".", 1)[0]
    return clean


def is_forbidden_node(node_identifier: str) -> bool:
    """Return True if node matches any forbidden node hostname, IP, or nickname."""
    raw_clean = str(node_identifier).strip().lower()
    normalized = normalize_node_name(node_identifier)

    if raw_clean in FORBIDDEN_IDENTIFIERS or normalized in FORBIDDEN_IDENTIFIERS:
        return True

    for fn in FORBIDDEN_SUBSTRINGS:
        if fn in raw_clean or fn in normalized:
            return True

    return False


def normalize_node_list(node_list: Union[str, List[str]]) -> List[str]:
    """Parse comma-separated strings or list of node identifiers."""
    if isinstance(node_list, str):
        nodes = [n.strip() for n in node_list.split(",") if n.strip()]
    else:
        nodes = [str(n).strip() for n in node_list if str(n).strip()]
    return nodes


def assert_nodes_allowed(node_list: Union[str, List[str]], allow_2node: bool = False) -> bool:
    """Assert that the given node list adheres to cluster isolation boundaries.

    Args:
        node_list: List or comma-separated string of node nicknames, hostnames, or IPs.
        allow_2node: Whether 2-node validation mode (node6 + node1) is permitted.

    Raises:
        NodeIsolationViolationError: If forbidden nodes (5, 7, 8, 9) are targeted,
            if node1 is targeted in 1-node mode, or if unknown nodes are requested.
    """
    nodes = normalize_node_list(node_list)
    if not nodes:
        raise NodeIsolationViolationError("Empty node list provided.")

    # 1. Strictly forbid nodes 5, 7, 8, 9
    for node in nodes:
        if is_forbidden_node(node):
            raise NodeIsolationViolationError(
                f"Node isolation violation: target node '{node}' is strictly forbidden. "
                f"Nodes 5, 7, 8, and 9 must remain untouched under all circumstances."
            )

    clean_nodes = [normalize_node_name(n) for n in nodes]

    for raw, clean in zip(nodes, clean_nodes):
        if is_forbidden_node(clean):
            raise NodeIsolationViolationError(
                f"Node isolation violation: target node '{raw}' is strictly forbidden. "
                f"Nodes 5, 7, 8, and 9 must remain untouched under all circumstances."
            )

    # 2. Check 1-node vs 2-node boundary
    if not allow_2node:
        for raw, clean in zip(nodes, clean_nodes):
            if clean in SECONDARY_IDENTIFIERS:
                raise NodeIsolationViolationError(
                    f"Node isolation violation: node1 ('{raw}') is permitted only for 2-node validation "
                    f"when allow_2node=True. Primary development must target node6 exclusively."
                )
            if clean not in PRIMARY_IDENTIFIERS:
                raise NodeIsolationViolationError(
                    f"Node isolation violation: node '{raw}' is not authorized. "
                    f"Only node6 (192.168.0.98) is allowed for 1-node execution."
                )
    else:
        # 2-node mode: only node6 + node1 allowed
        allowed_all = PRIMARY_IDENTIFIERS | SECONDARY_IDENTIFIERS
        for raw, clean in zip(nodes, clean_nodes):
            if clean not in allowed_all:
                raise NodeIsolationViolationError(
                    f"Node isolation violation: node '{raw}' is not authorized for 2-node validation. "
                    f"Only node6 (primary) and node1 (secondary) are permitted."
                )

        has_primary = any(clean in PRIMARY_IDENTIFIERS for clean in clean_nodes)
        if not has_primary:
            raise NodeIsolationViolationError(
                "2-node execution requires node6 as the primary development machine."
            )

    return True


def parse_npu_smi_proc_mem(output: str) -> Dict[str, Any]:
    """Parse output from `npu-smi info -t proc-mem -i <dev>`.

    Authoritative idle indicator:
    - Reports idle if output contains 'No process in device.' or 'No running process'.
    - Reports busy if any 'Process id' or memory footprint is detected.
    - Fails closed: an empty or unparsed output reports idle: False.
    """
    clean = output.strip()
    if "No process in device." in clean or "No running process" in clean:
        return {"active_processes": 0, "used_memory_bytes": 0, "idle": True}

    process_ids = re.findall(r"Process\s*id\s*:\s*(\d+)", clean, re.IGNORECASE)
    mem_matches = re.findall(r"(\d+)\s*(?:MB|MiB|GB|GiB)", clean, re.IGNORECASE)
    total_bytes = sum(int(m) * 1024 * 1024 for m in mem_matches)

    active = max(len(process_ids), len(mem_matches), 1 if ("Process id" in clean) else 0)
    return {
        "active_processes": active,
        "used_memory_bytes": total_bytes,
        "idle": False,
    }


def check_node1_idle(
    runner: Optional[Callable[[str], Tuple[str, str, int]]] = None,
    num_devices: int = 8,
    check_pgrep: bool = True,
) -> Tuple[bool, str]:
    """Verify that node1 is completely idle using proc-mem and pgrep.

    CRITICAL: Never relies on AICore% readings from npu-smi, which read 89-99%
    due to a known driver bug (Driver 25.2.1) even when only dead processes remain.

    Args:
        runner: Command executor taking a shell command string and returning (stdout, stderr, exit_code).
        num_devices: Number of NPU devices to check (0..num_devices-1).
        check_pgrep: Whether to inspect container and host process tables via pgrep.

    Returns:
        (is_idle, reason_or_detail)
    """
    def _default_runner(cmd: str) -> Tuple[str, str, int]:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return res.stdout, res.stderr, res.returncode

    run_cmd = runner if runner is not None else _default_runner

    # 1. Device process memory table check across devices 0..7
    for dev_id in range(num_devices):
        cmd = f"npu-smi info -t proc-mem -i {dev_id}"
        stdout, stderr, code = run_cmd(cmd)
        if code != 0 or not stdout.strip() or ("error" in (stderr or "").lower()):
            err_detail = stderr.strip() if (stderr and stderr.strip()) else ("empty output" if not stdout.strip() else f"exit code {code}")
            return False, f"Failed to query device on node1: exit code {code}, {err_detail}"
        parsed = parse_npu_smi_proc_mem(stdout)
        if not parsed["idle"] or "Process id" in stdout:
            return False, f"Device {dev_id} on node1 is busy: active process detected in device memory table."

    # 2. Process table inspection (pgrep)
    if check_pgrep:
        # Check for container or python training jobs
        pgrep_cmd = "pgrep -cf 'train_sft|precompute_latents|torchrun'"
        stdout, stderr, code = run_cmd(pgrep_cmd)
        try:
            count = int(stdout.strip())
            if count > 0:
                return False, f"node1 is busy: {count} training/precomputation process(es) running in process table."
        except ValueError:
            # If pgrep returns no matches, exit code is 1 and count is empty
            pass

    return True, "node1 is verified idle across all devices (0 active processes in proc-mem)."


def assert_current_host_allowed(allow_secondary: bool = False) -> None:
    """Assert that the script is currently running on an authorized machine.

    Raises:
        NodeIsolationViolationError: If running on node5, 7, 8, or 9, or on node1 without allow_secondary,
            or if the host is not positively authorized.
    """
    try:
        hostname = socket.gethostname().lower()
    except Exception:
        hostname = "unknown"

    local_ips = []
    try:
        _, _, ips = socket.gethostbyname_ex(hostname)
        local_ips.extend(ips)
    except Exception:
        pass

    raw_identifiers = [hostname] + local_ips
    all_identifiers: List[str] = []
    for ident in raw_identifiers:
        if ident:
            all_identifiers.append(ident)
            norm = normalize_node_name(ident)
            if norm and norm not in all_identifiers:
                all_identifiers.append(norm)

    # 1. Strictly forbid nodes 5, 7, 8, 9
    for ident in all_identifiers:
        if is_forbidden_node(ident):
            raise NodeIsolationViolationError(
                f"Current execution host '{ident}' is a FORBIDDEN node (node 5, 7, 8, or 9). "
                f"Jobs must never be launched on this machine."
            )

    # 2. Check secondary node isolation
    if not allow_secondary:
        for ident in all_identifiers:
            norm = normalize_node_name(ident)
            if ident in SECONDARY_IDENTIFIERS or norm in SECONDARY_IDENTIFIERS or "node1" in ident or "192.168.0.46" in ident:
                raise NodeIsolationViolationError(
                    f"Current execution host '{ident}' is node1, but allow_secondary is False. "
                    f"Primary development must take place exclusively on node6."
                )

    # 3. Positive verification of authorization
    allowed_set = (PRIMARY_IDENTIFIERS | SECONDARY_IDENTIFIERS) if allow_secondary else PRIMARY_IDENTIFIERS
    is_authorized = any(ident in allowed_set for ident in all_identifiers)
    if not is_authorized:
        target_role = "primary (node6) or secondary (node1)" if allow_secondary else "primary (node6)"
        raise NodeIsolationViolationError(
            f"Current execution host '{hostname}' (identifiers: {sorted(set(all_identifiers))}) "
            f"is not an authorized {target_role} cluster host."
        )
