#!/usr/bin/env python3
"""CLI utility to verify Ascend cluster node isolation and node1 idleness.

Usage:
  python check_node_isolation.py --nodes node6
  python check_node_isolation.py --nodes node6,node1 --allow-2node --check-idle
"""

import argparse
import sys
from typing import Tuple

try:
    from colossalai.cluster.node_guard import (
        NodeIsolationViolationError,
        assert_nodes_allowed,
        check_node1_idle,
        normalize_node_list,
    )
except ImportError:
    # If running directly from script directory
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from colossalai.cluster.node_guard import (
        NodeIsolationViolationError,
        assert_nodes_allowed,
        check_node1_idle,
        normalize_node_list,
    )


def main():
    parser = argparse.ArgumentParser(description="Ascend Cluster Node Isolation Enforcement Guardrail.")
    parser.add_argument(
        "--nodes",
        type=str,
        default="node6",
        help="Comma-separated list of target nodes, hostnames, or IPs (e.g. 'node6' or 'node6,node1').",
    )
    parser.add_argument(
        "--allow-2node",
        action="store_true",
        help="Allow 2-node validation configuration (node6 primary + node1 secondary).",
    )
    parser.add_argument(
        "--check-idle",
        action="store_true",
        help="Verify node1 device process memory (npu-smi) and process table (pgrep) are completely idle.",
    )
    parser.add_argument(
        "--mock-idle",
        action="store_true",
        help="Mock idle responses for offline/CI environments.",
    )
    parser.add_argument(
        "--mock-busy",
        action="store_true",
        help="Mock busy device responses for testing failure paths.",
    )

    args = parser.parse_args()
    nodes = normalize_node_list(args.nodes)

    # 1. Assert nodes allowed
    try:
        assert_nodes_allowed(nodes, allow_2node=args.allow_2node)
        print(f"[OK] Node isolation check PASSED for nodes: {nodes}")
    except NodeIsolationViolationError as e:
        print(f"[ERROR] Node isolation VIOLATION: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Check idle if requested or if node1 is targeted
    target_has_node1 = any("node1" in n.lower() or "192.168.0.46" in n for n in nodes)
    if args.check_idle or (target_has_node1 and (args.mock_idle or args.mock_busy)):
        mock_runner = None
        if args.mock_busy:
            mock_runner = lambda cmd: ("Process id: 12345 (train_sft.py) | 16384 MB", "", 0)
        elif args.mock_idle:
            mock_runner = lambda cmd: ("npu-smi 23.0.0\nNo process in device.\n", "", 0)

        is_idle, reason = check_node1_idle(runner=mock_runner)
        if not is_idle:
            print(f"[ERROR] Node 1 idle pre-flight check FAILED: {reason}", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] Node 1 idle pre-flight check PASSED: {reason}")

    print("[SUCCESS] All node isolation constraints verified successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
