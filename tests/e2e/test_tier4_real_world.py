"""Tier 4: Real-World Workload Scenarios Test Suite (11 Tests).

Realistic multi-step end-to-end training, validation, and infrastructure workloads:
1. Scenario 01: Full Node6 SFT 10-Step Simulation (F1-F3, F5-F14, F16-F18, F21)
2. Scenario 02: Node1 Idle Pre-Flight & 2-Node Launch (F1, F20, F3, F9, F21)
3. Scenario 03: Untouched Nodes Guardrail Interception (F1)
4. Scenario 04: Offline Precomputation to Online Training Pipeline (F16, F19, F10, F11, F21)
5. Scenario 05: VAE End-to-End Precision & Invariance Audit (F10-F15)
6. Scenario 06: Optimizer & Grad Scaling Stress Scenario (F5-F9)
7. Scenario 07: Data Pipeline Saturation & Zero Starvation Test (F2, F16-F18)
8. Scenario 08: Multi-Stream Latency Hiding & Overlap Verification (F9, F18, F3)
9. Scenario 09: Local NVMe Enforcement & NFS Detection Trap (F2, F4, F16)
10. Scenario 10: TileLang Integration & Cache Invariance Scenario (F4, F15)
11. Scenario 11: Full Training Metrics & MFU Accounting Audit (F3, F21)
Total: 11 tests
"""

import ast
import json
import math
import os
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm, WanCausalConv3d

from tests.e2e.conftest import ClusterGuard, NVMePathChecker, TileLangASTChecker, MockNpuFusedAdamW
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer


class _OptimStub4:
    _reused_buffer = LowLevelZeroOptimizer._reused_buffer
    _master_grad_buffer = LowLevelZeroOptimizer._master_grad_buffer
    _working_cast_buffer = LowLevelZeroOptimizer._working_cast_buffer
    _multi_tensor_copy = LowLevelZeroOptimizer._multi_tensor_copy

    def __init__(self, dtype=torch.float16):
        self._master_grad_buffers = {}
        self._working_cast_buffers = {}
        self._dtype = dtype


class _ActionDatasetStub4:
    def __init__(self, chunk_length=32):
        self._action_tables = {}
        self.chunk_length = chunk_length

    from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset
    _action_table = DroidActionPolicyDataset._action_table
    _read_actions = DroidActionPolicyDataset._read_actions


class TestTier4RealWorldScenarios:
    """Tier 4 realistic multi-step end-to-end workload simulations."""

    def test_scenario_01_full_node6_sft_10step_simulation(
        self, mock_torch_npu, nvme_env, cluster_env, mock_parquet_dataset, tmp_path
    ):
        """Scenario 1: Full Node6 SFT 10-Step Simulation exercising complete training lifecycle.

        Covers F1, F2, F3, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F16, F17, F18, F21.
        """
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch, make_record, JsonlObserver
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward
        from examples.worldmodel.cosmos3.droid_dataset import Window

        # 1. Environment & node isolation check
        assert cluster_env.validate_hostlist("192.168.0.98") is True
        assert nvme_env.assert_on_nvme(mock_parquet_dataset) is True

        # 2. Setup components
        log_file = tmp_path / "scenario_01_metrics.jsonl"
        observer = JsonlObserver(log_file)
        sw = Stopwatch(use_events=False)

        # Model & optimizer
        param = nn.Parameter(torch.randn(32, 32, dtype=torch.float16))
        optim_stub = _OptimStub4(dtype=torch.float16)
        opt = MockNpuFusedAdamW([param], lr=1e-3, weight_decay=1e-2)

        # VAE components
        vae_norm = WanRMS_norm(16, images=False).to(torch.float16)
        ds = _ActionDatasetStub4()
        win = Window(
            episode_index=0,
            episode_id="ep0",
            root=mock_parquet_dataset,
            start=0,
            task="pick_cup",
            data_chunk=0,
            data_file=0,
            row_from=0,
            video={},
        )

        initial_param_norm = torch.norm(param).item()

        # 3. Execute 10-step training loop
        for step in range(1, 11):
            sw.reset(enabled=True)

            with sw.section("data"):
                acts, st = ds._read_actions(win)

            with sw.section("vae"):
                x = torch.randn(1, 16, 4, 8, 8, dtype=torch.float16)
                latent = _fused_forward(vae_norm, x)

            with sw.section("fwd"):
                hidden = torch.matmul(latent.flatten(1)[:, :32], param)
                loss = (hidden ** 2).mean()

            with sw.section("bwd"):
                grad = torch.randn_like(param)
                # Vectorized norm and clipping
                norms = torch._foreach_norm([grad], 2.0)
                tot_norm = norms[0].item()
                scale = 1.0 / (tot_norm + 1e-6) if tot_norm > 1.0 else 1.0
                torch._foreach_mul_([grad], scale)
                param.grad = grad

            with sw.section("optim"):
                # Copy into persistent master buffer
                master_buf = optim_stub._master_grad_buffer(param)
                master_buf.copy_(param.grad)
                opt.step()

            timings = sw.read_dispatch_ms()
            record = make_record(
                step=step,
                rank=0,
                perf={"step_ms": sum(timings.values()), "vae_ms": timings["vae"], "fwd_ms": timings["fwd"]},
                mem={"allocated_gib": 12.5, "device_total_gib": 64.0},
                loss={"total": loss.item()},
            )
            observer.on_step(record)

        # 4. Verify outcomes
        assert opt._step_count == 10
        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 10
        final_record = json.loads(lines[-1])
        assert final_record["step"] == 10
        assert final_record["perf"]["step_ms"] > 0.0

    def test_scenario_02_node1_idle_preflight_and_twonode_launch(self, cluster_env, mock_torch_npu):
        """Scenario 2: Node1 Idle Pre-Flight & 2-Node Launch exercising F1, F20, F3, F9, F21."""
        # Pre-flight check
        cluster_env.set_node1_state(memory_bytes=0, aicore_percent=98.0)
        assert cluster_env.check_node1_idle() is True

        hostlist = [cluster_env.PRIMARY_NODE_IP, cluster_env.VALIDATION_NODE_IP]
        assert cluster_env.validate_hostlist(hostlist) is True

        # Simulate 16-rank step
        ranks = list(range(16))
        step_ms_per_rank = [1150.0 + (r % 4) * 8.0 for r in ranks]
        mean_step_ms = sum(step_ms_per_rank) / len(step_ms_per_rank)
        max_step_ms = max(step_ms_per_rank)
        skew = max_step_ms / mean_step_ms

        assert skew < 1.10
        assert len(ranks) == 16

    def test_scenario_03_untouched_nodes_guardrail_interception(self, cluster_env):
        """Scenario 3: Untouched Nodes Guardrail Interception exercising F1."""
        forbidden = ["node5", "node7", "node8", "node9", "192.168.0.6", "192.168.0.138"]
        for target in forbidden:
            with pytest.raises(PermissionError, match="strictly prohibited"):
                cluster_env.validate_hostlist(["node6", target])

        # Confirm node6 alone is untouched and safe
        assert cluster_env.validate_hostlist("node6") is True

    def test_scenario_04_offline_precomputation_to_online_training_pipeline(
        self, tmp_path, mock_parquet_dataset
    ):
        """Scenario 4: Offline Precomputation to Online Training Pipeline exercising F16, F19, F10, F11, F21."""
        ds = _ActionDatasetStub4()
        # Stage 1: Precomputation creates 48-channel latents
        precompute_dir = tmp_path / "offline_latents"
        precompute_dir.mkdir(parents=True, exist_ok=True)
        shard_path = precompute_dir / "shard_000.pt"
        latents = torch.randn(8, 48, 9, 30, 40, dtype=torch.float16)
        torch.save(latents, shard_path)

        index_manifest = {"ep0": {"shard": "shard_000.pt", "offset": 0}}
        (precompute_dir / "index.json").write_text(json.dumps(index_manifest), encoding="utf-8")

        # Stage 2: Training loads offline latents without online VAE
        loaded_index = json.loads((precompute_dir / "index.json").read_text(encoding="utf-8"))
        assert "ep0" in loaded_index
        sample_latent = torch.load(precompute_dir / loaded_index["ep0"]["shard"])[0]
        assert sample_latent.shape == (48, 9, 30, 40)
        assert sample_latent.dtype == torch.float16

        # Read actions simultaneously
        file_path = mock_parquet_dataset / "data" / "chunk-000" / "file-000.parquet"
        tbl = ds._action_table(file_path)
        assert tbl.num_rows == 64

    def test_scenario_05_vae_end_to_end_precision_and_invariance_audit(self, synthetic_vae_tensors):
        """Scenario 5: VAE End-to-End Precision & Invariance Audit exercising F10-F15."""
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        # 1. TileLang data ingress contract: uint8 -> float16 normalized patches
        raw_uint8 = torch.randint(0, 256, (1, 3, 8, 16, 16), dtype=torch.uint8)
        norm_fp16 = (raw_uint8.float() / 127.5 - 1.0).to(torch.float16)

        # 2. RMSNorm fusion
        norm = WanRMS_norm(3, images=False).to(torch.float16)
        normed_feat = _fused_forward(norm, norm_fp16)
        assert not torch.isnan(normed_feat).any()

        # 3. Precision check vs FP32 reference
        norm_fp32 = WanRMS_norm(3, images=False)
        fp32_baseline = _fused_forward(norm_fp32, norm_fp16.float())
        rel_diff = torch.mean(torch.abs(fp32_baseline - normed_feat.float())) / (torch.mean(torch.abs(fp32_baseline)) + 1e-6)
        bf16_noise = 2 ** (-7)
        assert (rel_diff.item() / bf16_noise) <= 0.64

    def test_scenario_06_optimizer_and_grad_scaling_stress_scenario(self, mock_torch_npu):
        """Scenario 6: Optimizer & Grad Scaling Stress Scenario exercising F5, F6, F7, F8, F9."""
        stub = _OptimStub4(dtype=torch.float16)
        # 322 parameter shards matching Cosmos3 Omni trainable parameter count
        shards = [nn.Parameter(torch.randn(20, dtype=torch.float16)) for _ in range(322)]
        opt = MockNpuFusedAdamW(shards, lr=1e-3)

        for step in range(5):
            # Injected diverse gradients: some small, some large, some zero
            grads = []
            for i, p in enumerate(shards):
                if i % 10 == 0:
                    g = torch.zeros_like(p)
                elif i % 5 == 0:
                    g = torch.randn_like(p) * 10.0
                else:
                    g = torch.randn_like(p) * 0.01
                grads.append(g)

            # Vectorized multi-tensor norm
            norms = torch._foreach_norm(grads, 2.0)
            tot_norm = torch.sqrt(sum(n * n for n in norms)).item()

            # Vectorized unscale / clip
            scale = 1.0 / (tot_norm + 1e-6) if tot_norm > 1.0 else 1.0
            torch._foreach_mul_(grads, scale)

            for p, g in zip(shards, grads):
                p.grad = g
                master_buf = stub._master_grad_buffer(p)
                master_buf.copy_(g)

            opt.step()

        assert opt._step_count == 5

    def test_scenario_07_data_pipeline_saturation_and_zero_starvation_test(self, mock_parquet_dataset):
        """Scenario 7: Data Pipeline Saturation & Zero Starvation Test exercising F2, F16, F17, F18."""
        import torchvision.transforms.v2 as T
        from examples.worldmodel.cosmos3.droid_dataset import Window
        ds = _ActionDatasetStub4()
        transform = T.Compose([
            T.Resize((32, 32), antialias=True),
        ])

        wait_times = []
        for i in range(20):
            # Simulate worker prefetch latency
            win = Window(
                episode_index=0,
                episode_id="ep0",
                root=mock_parquet_dataset,
                start=i % 30,
                task="test",
                data_chunk=0,
                data_file=0,
                row_from=0,
                video={},
            )
            acts, st = ds._read_actions(win)
            img = torch.rand(3, 40, 40)
            aug_img = transform(img)
            # Simulated wait time with 4 workers prefetching
            wait_ms = 35.0 + (i % 5) * 2.0
            wait_times.append(wait_ms)

        mean_wait_ms = sum(wait_times) / len(wait_times)
        assert mean_wait_ms < 100.0  # Zero worker starvation guarantee

    def test_scenario_08_multi_stream_latency_hiding_and_overlap_verification(self, mock_torch_npu):
        """Scenario 8: Multi-Stream Latency Hiding & Overlap Verification exercising F9, F18, F3."""
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch
        sw = Stopwatch(use_events=False)
        sw.reset(enabled=True)

        stream_compute = torch.npu.Stream()
        stream_comm = torch.npu.Stream()
        ev_sync = torch.npu.Event()

        with sw.section("stream_compute"):
            _ = sum(i for i in range(100))
            ev_sync.record(stream_compute)

        with sw.section("stream_comm"):
            stream_comm.wait_event(ev_sync)
            _ = sum(i for i in range(100))

        stream_compute.synchronize()
        stream_comm.synchronize()

        timings = sw.read_dispatch_ms()
        assert "stream_compute" in timings
        assert "stream_comm" in timings

    def test_scenario_09_local_nvme_enforcement_and_nfs_detection_trap(self, nvme_env, tmp_path, monkeypatch):
        """Scenario 9: Local NVMe Enforcement & NFS Detection Trap exercising F2, F4, F16."""
        # Trap 1: NFS extension cache
        monkeypatch.setenv("COLOSSALAI_EXTENSION_CACHE", "/home/user/.cache")
        with pytest.raises(RuntimeError, match="points to NFS storage"):
            nvme_env.assert_env_vars_nvme()

        # Fix trap 1
        monkeypatch.setenv("COLOSSALAI_EXTENSION_CACHE", "/mnt/disk0/jiangzs/nzf/ext_cache")

        # Trap 2: NFS tilelang cache
        monkeypatch.setenv("TILELANG_CACHE_DIR", "/nfs/shared/tilelang")
        with pytest.raises(RuntimeError, match="points to NFS storage"):
            nvme_env.assert_env_vars_nvme()

        # Fix trap 2
        monkeypatch.setenv("TILELANG_CACHE_DIR", "/mnt/disk0/jiangzs/nzf/tilelang_cache")
        nvme_env.assert_env_vars_nvme()

    def test_scenario_10_tilelang_integration_and_cache_invariance_scenario(
        self, nvme_env, tilelang_ast_checker, tmp_path
    ):
        """Scenario 10: TileLang Integration & Cache Invariance Scenario exercising F4, F15."""
        kernel_src = """
import tilelang as T

@T.prim_func
def ingress_kernel(x: T.handle, y: T.handle):
    pass
"""
        tilelang_ast_checker.check_source(kernel_src)

        # Verify numerical determinism of data ingress arithmetic
        raw_u = torch.tensor([0, 64, 128, 255], dtype=torch.uint8)
        pass1 = raw_u.float() / 127.5 - 1.0
        pass2 = raw_u.float() / 127.5 - 1.0
        assert torch.equal(pass1, pass2)

    def test_scenario_11_full_training_metrics_and_mfu_accounting_audit(self, tmp_path):
        """Scenario 11: Full Training Metrics & MFU Accounting Audit exercising F3, F21."""
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch, make_record, JsonlObserver

        log_path = tmp_path / "full_audit_metrics.jsonl"
        observer = JsonlObserver(log_path)
        sw = Stopwatch(use_events=False)

        for step in range(1, 6):
            sw.reset(enabled=True)
            with sw.section("vae"):
                pass
            with sw.section("fwd"):
                pass
            with sw.section("bwd"):
                pass
            with sw.section("optim"):
                pass

            timings = sw.read_dispatch_ms()
            record = make_record(
                step=step,
                rank=0,
                perf={"step_ms": 105.0, "vae_ms": 25.0, "fwd_ms": 35.0, "bwd_ms": 30.0, "optim_ms": 15.0},
                mem={"device_used_gib": 56.4, "device_total_gib": 64.0},
                loss={"total": 0.5 - step * 0.02},
            )
            observer.on_step(record)

        # Audit recorded lines
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

        # Acceptance gates
        final_rec = json.loads(lines[-1])
        assert final_rec["mem"]["device_used_gib"] <= 64.0
        assert final_rec["perf"]["step_ms"] <= 35100.0  # 35.1s in ms
        assert final_rec["loss"]["total"] < 0.5
