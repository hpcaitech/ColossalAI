"""Shared fixtures and mocks for Cosmos3 Ascend 910B2 E2E test suite.

Provides opaque-box mock environments for:
- Ascend 910B2 NPU cluster topology (node6 primary, node1 validation, node 5/7/8/9 forbidden)
- torch_npu and NpuFusedAdamW off-cluster emulation
- NVMe vs NFS storage path verification
- TileLang AST guardrails against `from __future__ import annotations`
- Synthetic datasets and VAE tensors for end-to-end pipelines
"""

import ast
import math
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==============================================================================
# 1. Cluster Guard & Topology Registry
# ==============================================================================


class ClusterGuard:
    """Enforces strict Ascend 910B2 cluster node isolation and validation rules."""

    PRIMARY_NODE_NAME = "node6"
    PRIMARY_NODE_IP = "192.168.0.98"

    VALIDATION_NODE_NAME = "node1"
    VALIDATION_NODE_IP = "192.168.0.46"

    FORBIDDEN_NODES = {
        "node5": "192.168.0.6",
        "node7": "192.168.0.138",
        "node8": "192.168.0.114",
        "node9": "192.168.0.88",
    }

    FORBIDDEN_IDENTIFIERS = set(FORBIDDEN_NODES.keys()) | set(FORBIDDEN_NODES.values())
    ALLOWED_IDENTIFIERS = {
        PRIMARY_NODE_NAME,
        PRIMARY_NODE_IP,
        VALIDATION_NODE_NAME,
        VALIDATION_NODE_IP,
        "localhost",
        "127.0.0.1",
    }

    def __init__(self):
        self.node1_memory_bytes: int = 0
        self.node1_aicore_util: float = 0.0

    def is_forbidden_node(self, identifier: str) -> bool:
        clean = identifier.strip().lower()
        return clean in self.FORBIDDEN_IDENTIFIERS or any(fn in clean for fn in self.FORBIDDEN_NODES.keys())

    def validate_hostlist(self, hostlist: Union[str, List[str]]) -> bool:
        """Validates that a proposed hostlist touches ONLY allowed nodes."""
        if isinstance(hostlist, str):
            hosts = [h.strip() for h in hostlist.split(",") if h.strip()]
        else:
            hosts = list(hostlist)

        for host in hosts:
            if self.is_forbidden_node(host):
                raise PermissionError(
                    f"Node isolation violation: access to forbidden node '{host}' is strictly prohibited. "
                    f"Nodes 5, 7, 8, and 9 must remain untouched."
                )

        # Ensure node6 is included for any multi-node run
        clean_hosts = [h.strip().lower() for h in hosts]
        if len(clean_hosts) > 1:
            has_primary = self.PRIMARY_NODE_NAME in clean_hosts or self.PRIMARY_NODE_IP in clean_hosts
            if not has_primary:
                raise ValueError("Multi-node runs must include node6 as primary host.")

        return True

    def set_node1_state(self, memory_bytes: int, aicore_percent: float = 0.0):
        self.node1_memory_bytes = memory_bytes
        self.node1_aicore_util = aicore_percent

    def check_node1_idle(self, memory_bytes: Optional[int] = None, aicore_percent: Optional[float] = None) -> bool:
        """Determines if node1 is truly idle.

        Important cluster observation: npu-smi AICore% can report 93-99% on idle nodes
        due to zombie counters, but npu-smi info -t proc-mem reports 'No process in device.'
        Device memory occupancy is the authoritative ground truth.
        """
        mem = self.node1_memory_bytes if memory_bytes is None else memory_bytes
        if mem > 0:
            raise RuntimeError(
                f"Node1 pre-flight validation failed: active device memory occupancy detected "
                f"({mem / (1024**3):.2f} GiB). Node1 must have 0 memory in use."
            )
        return True

    def parse_npu_smi_proc_mem(self, output: str) -> Dict[str, Any]:
        """Parses npu-smi info -t proc-mem CLI stdout."""
        clean = output.strip()
        if "No process in device." in clean or "No running process" in clean:
            return {"active_processes": 0, "used_memory_bytes": 0, "idle": True}

        # Match memory lines: e.g. "python | 12345 | 14500 MB"
        mem_matches = re.findall(r"(\d+)\s*(?:MB|MiB|GB|GiB)", clean, re.IGNORECASE)
        total_bytes = 0
        for m in mem_matches:
            total_bytes += int(m) * 1024 * 1024

        procs = len(mem_matches)
        return {
            "active_processes": procs,
            "used_memory_bytes": total_bytes,
            "idle": procs == 0 and total_bytes == 0,
        }


# ==============================================================================
# 2. NVMe vs NFS Storage Inspector
# ==============================================================================


class NVMePathChecker:
    """Verifies that execution directories and caches strictly reside on local NVMe."""

    NVME_ROOT_PREFIX = "/mnt/disk0"
    FORBIDDEN_NFS_PATTERNS = ["/home/", "/nfs/", "/mnt/nfs/", "nfs://", "/shared/"]

    @classmethod
    def is_nfs_path(cls, path: Union[str, Path]) -> bool:
        p_str = str(path).strip()
        for pattern in cls.FORBIDDEN_NFS_PATTERNS:
            if p_str.startswith(pattern) or pattern in p_str:
                return True
        return False

    @classmethod
    def assert_on_nvme(cls, path: Union[str, Path], allow_temp: bool = True) -> bool:
        p = Path(path).resolve()
        p_str = str(p)

        if cls.is_nfs_path(p_str):
            raise RuntimeError(
                f"NFS storage violation: path '{p_str}' resides on NFS. "
                f"All caches, environments, and repos must reside on local NVMe to avoid "
                f"FileBaton deadlock and dataloader starvation."
            )

        if allow_temp and (p_str.startswith("/tmp") or p_str.startswith("/private/tmp") or "pytest" in p_str):
            return True

        if not (p_str.startswith(cls.NVME_ROOT_PREFIX) or p_str.startswith("/tmp") or allow_temp):
            raise RuntimeError(f"Path '{p_str}' is not on local NVMe mount ({cls.NVME_ROOT_PREFIX}).")
        return True

    @classmethod
    def assert_env_vars_nvme(cls) -> None:
        cache_env = os.environ.get("COLOSSALAI_EXTENSION_CACHE")
        tilelang_env = os.environ.get("TILELANG_CACHE_DIR")

        if cache_env and cls.is_nfs_path(cache_env):
            raise RuntimeError(f"COLOSSALAI_EXTENSION_CACHE '{cache_env}' points to NFS storage.")
        if tilelang_env and cls.is_nfs_path(tilelang_env):
            raise RuntimeError(f"TILELANG_CACHE_DIR '{tilelang_env}' points to NFS storage.")


# ==============================================================================
# 3. TileLang AST Checker
# ==============================================================================


class TileLangASTChecker:
    """Enforces no `from __future__ import annotations` in TileLang kernels."""

    @staticmethod
    def check_source(source_code: str) -> None:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                for alias in node.names:
                    if alias.name == "annotations":
                        raise ValueError(
                            "TileLang AST violation: 'from __future__ import annotations' is strictly "
                            "prohibited in TileLang kernels. Postponing type evaluation causes TVM "
                            "kTVMObjectHandle parsing failures (expected Object but got str)."
                        )

    @classmethod
    def check_file(cls, file_path: Union[str, Path]) -> None:
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Source file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            cls.check_source(f.read())


# ==============================================================================
# 4. Mock Torch NPU & NpuFusedAdamW Emulation
# ==============================================================================


class MockNPUEvent:
    """Mock implementation of torch.npu.Event with timing support."""

    def __init__(self, enable_timing: bool = True):
        self.enable_timing = enable_timing
        self._record_time: Optional[float] = None

    def record(self, stream=None):
        import time

        self._record_time = time.perf_counter()

    def elapsed_time(self, end_event: "MockNPUEvent") -> float:
        """Returns elapsed time in milliseconds."""
        if self._record_time is None or end_event._record_time is None:
            return 1.25  # Nominal baseline duration in ms
        return max(0.001, (end_event._record_time - self._record_time) * 1000.0)

    def synchronize(self):
        pass


class MockNPUStream:
    """Mock implementation of torch.npu.Stream."""

    def __init__(self, device=None):
        self.device = device

    def wait_event(self, event):
        pass

    def synchronize(self):
        pass


class MockNpuFusedAdamW(torch.optim.Optimizer):
    """High-fidelity emulation of torch_npu.optim.NpuFusedAdamW.

    Preserves exact AdamW update equations and simulates combined tensor buffers.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.combined_buffers_initialized: bool = False
        self._param_ptrs: Dict[int, int] = {}
        self._step_count: int = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Decay the first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        self.combined_buffers_initialized = True
        return loss


def create_mock_torch_npu():
    """Builds a mock `torch_npu` module for off-cluster portability."""
    mod = types.ModuleType("torch_npu")
    optim_mod = types.ModuleType("torch_npu.optim")
    optim_mod.NpuFusedAdamW = MockNpuFusedAdamW
    mod.optim = optim_mod

    # Add torch.npu emulation
    class NPUNamespace:
        Event = MockNPUEvent
        Stream = MockNPUStream

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 8

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def set_device(device):
            pass

        @staticmethod
        def synchronize(device=None):
            pass

        @staticmethod
        def mem_get_info(device=None) -> Tuple[int, int]:
            # 64 GiB Ascend 910B2 capacity
            total = 64 * 1024 * 1024 * 1024
            free = 58 * 1024 * 1024 * 1024
            return free, total

        @staticmethod
        def memory_allocated(device=None) -> int:
            return 6 * 1024 * 1024 * 1024

        @staticmethod
        def max_memory_allocated(device=None) -> int:
            return 8 * 1024 * 1024 * 1024

        @staticmethod
        def memory_reserved(device=None) -> int:
            return 10 * 1024 * 1024 * 1024

        @staticmethod
        def max_memory_reserved(device=None) -> int:
            return 12 * 1024 * 1024 * 1024

        @staticmethod
        def empty_cache():
            pass

    mod.npu = NPUNamespace
    return mod


# ==============================================================================
# 5. Pytest Fixtures
# ==============================================================================


@pytest.fixture
def cluster_env() -> ClusterGuard:
    """Fixture providing Ascend cluster node isolation validation."""
    return ClusterGuard()


@pytest.fixture
def nvme_env() -> type[NVMePathChecker]:
    """Fixture providing NVMe vs NFS storage checks."""
    return NVMePathChecker


@pytest.fixture
def tilelang_ast_checker() -> type[TileLangASTChecker]:
    """Fixture providing TileLang AST annotations checker."""
    return TileLangASTChecker


@pytest.fixture
def mock_torch_npu(monkeypatch):
    """Fixture injecting MockTorchNPU into sys.modules and torch.npu."""
    mock_mod = create_mock_torch_npu()
    monkeypatch.setitem(sys.modules, "torch_npu", mock_mod)
    monkeypatch.setitem(sys.modules, "torch_npu.optim", mock_mod.optim)

    # Attach to torch.npu if absent
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", mock_mod.npu, raising=False)

    return mock_mod


@pytest.fixture
def synthetic_vae_tensors():
    """Generates synthetic Wan VAE video and latent tensors for testing."""
    B, C, T, H, W = 1, 3, 9, 32, 32
    video_tensor_fp32 = torch.randn(B, C, T, H, W, dtype=torch.float32)
    video_tensor_fp16 = video_tensor_fp32.clone().to(torch.float16)

    # Standard Wan VAE 48-channel latent output
    num_latent_t = (T - 1) // 4 + 1
    latent_tensor = torch.randn(B, 48, num_latent_t, H // 16, W // 16, dtype=torch.float16)

    return {
        "video_fp32": video_tensor_fp32,
        "video_fp16": video_tensor_fp16,
        "latent_fp16": latent_tensor,
        "shape_video": (B, C, T, H, W),
        "shape_latent": (B, 48, num_latent_t, H // 16, W // 16),
    }


@pytest.fixture
def mock_parquet_dataset(tmp_path) -> Path:
    """Creates a minimal valid DROID Parquet action table for LRU cache testing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    num_rows = 64
    joint_action = [[0.01 * (i + j) for j in range(7)] for i in range(num_rows)]
    gripper_action = [[1.0 if i % 2 == 0 else 0.0] for i in range(num_rows)]
    state_joint = [[0.05 * (i + j) for j in range(7)] for i in range(num_rows)]
    state_gripper = [[0.8 if i % 2 == 0 else 0.2] for i in range(num_rows)]
    episode_index = [0] * num_rows

    table = pa.Table.from_arrays(
        [
            pa.array(joint_action),
            pa.array(gripper_action),
            pa.array(state_joint),
            pa.array(state_gripper),
            pa.array(episode_index),
        ],
        names=[
            "action.joint_position",
            "action.gripper_position",
            "observation.state.joint_positions",
            "observation.state.gripper_position",
            "episode_index",
        ],
    )

    data_dir = tmp_path / "droid_dataset"
    chunk_dir = data_dir / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    file_path = chunk_dir / "file-000.parquet"
    pq.write_table(table, file_path)

    # Create a second episode for multi-table caching tests
    file_path2 = chunk_dir / "file-001.parquet"
    pq.write_table(table, file_path2)

    return data_dir
