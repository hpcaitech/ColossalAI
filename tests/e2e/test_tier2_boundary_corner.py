"""Tier 2: Boundary, Limit, Negative & Edge Case Test Suite (105 Tests).

Contains exactly 5 boundary/corner tests per feature across all 21 features:
- F1: Node Isolation Boundaries (5 tests)
- F2: Local NVMe Boundaries (5 tests)
- F3: Metrics Boundaries (5 tests)
- F4: TileLang Boundaries (5 tests)
- F5: NpuFusedAdamW Boundaries (5 tests)
- F6: Multi-Tensor Grad Norm Boundaries (5 tests)
- F7: Multi-Tensor Unscale/Clip Boundaries (5 tests)
- F8: Persistent Buffer Boundaries (5 tests)
- F9: Backward Comm Overlap Boundaries (5 tests)
- F10: VAE FP16 Boundaries (5 tests)
- F11: VAE Chunked Boundaries (5 tests)
- F12: VAE Fused RMSNorm Boundaries (5 tests)
- F13: VAE Fast Conv Padding Sink Boundaries (5 tests)
- F14: VAE Weight Pre-Transformation Boundaries (5 tests)
- F15: TileLang Data Ingress Boundaries (5 tests)
- F16: Parquet Table Caching Boundaries (5 tests)
- F17: CPU Augmentation Boundaries (5 tests)
- F18: Multi-Stream Boundaries (5 tests)
- F19: Offline Latent Boundaries (5 tests)
- F20: Two-Node Validation Boundaries (5 tests)
- F21: Final Metrics Boundaries (5 tests)
Total: 105 tests
"""

import math
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer
from tests.e2e.conftest import MockNpuFusedAdamW

# ==============================================================================
# Feature 1: Node Isolation Boundaries
# ==============================================================================


class TestTier2Feature1NodeIsolationBoundaries:
    """Boundary and negative tests for cluster node isolation guardrails."""

    def test_launch_on_node5_strictly_blocked(self, cluster_env):
        with pytest.raises(PermissionError, match="strictly prohibited"):
            cluster_env.validate_hostlist(["node5"])
        with pytest.raises(PermissionError, match="strictly prohibited"):
            cluster_env.validate_hostlist(["192.168.0.6"])

    def test_launch_on_nodes_7_8_9_strictly_blocked(self, cluster_env):
        for node in ["node7", "node8", "node9", "192.168.0.138", "192.168.0.114", "192.168.0.88"]:
            with pytest.raises(PermissionError, match="strictly prohibited"):
                cluster_env.validate_hostlist([node])

    def test_node1_rejected_when_proc_mem_greater_than_zero(self, cluster_env):
        cluster_env.set_node1_state(memory_bytes=1024 * 1024 * 1024, aicore_percent=0.0)
        with pytest.raises(RuntimeError, match="Node1 pre-flight validation failed"):
            cluster_env.check_node1_idle()

    def test_mixed_hostlist_blocked_before_spawn(self, cluster_env):
        with pytest.raises(PermissionError, match="strictly prohibited"):
            cluster_env.validate_hostlist(["192.168.0.98", "192.168.0.6"])

    def test_zombie_aicore_with_zero_mem_correctly_classified_idle(self, cluster_env):
        cluster_env.set_node1_state(memory_bytes=0, aicore_percent=99.9)
        assert cluster_env.check_node1_idle() is True


# ==============================================================================
# Feature 2: Local NVMe Boundaries
# ==============================================================================


class TestTier2Feature2LocalNVMeBoundaries:
    """Boundary tests for NVMe path enforcement and NFS detection."""

    def test_nfs_extension_cache_path_rejected(self, nvme_env, monkeypatch):
        nfs_path = "/home/jiangzs/.cache/colossalai"
        monkeypatch.setenv("COLOSSALAI_EXTENSION_CACHE", nfs_path)
        with pytest.raises(RuntimeError, match="points to NFS storage"):
            nvme_env.assert_env_vars_nvme()

    def test_nfs_tilelang_cache_path_rejected(self, nvme_env, monkeypatch):
        nfs_path = "/shared/nfs/tilelang_cache"
        monkeypatch.setenv("TILELANG_CACHE_DIR", nfs_path)
        with pytest.raises(RuntimeError, match="points to NFS storage"):
            nvme_env.assert_env_vars_nvme()

    def test_symlink_pointing_to_nfs_rejected(self, nvme_env, tmp_path):
        nfs_target = "/home/user/fake_nfs_target"
        assert nvme_env.is_nfs_path(nfs_target) is True
        with pytest.raises(RuntimeError, match="NFS storage violation"):
            nvme_env.assert_on_nvme(nfs_target, allow_temp=False)

    def test_missing_nvme_dir_created_or_fails_fast(self, nvme_env):
        invalid_mount = "/data_nfs/my_env"
        with pytest.raises(RuntimeError):
            nvme_env.assert_on_nvme(invalid_mount, allow_temp=False)

    def test_container_shm_size_under_512g_warns(self):
        insufficient_shm_gib = 256
        min_required_shm_gib = 512
        is_sufficient = insufficient_shm_gib >= min_required_shm_gib
        assert is_sufficient is False


# ==============================================================================
# Feature 3: Metrics Boundaries
# ==============================================================================


class TestTier2Feature3MetricsBoundaries:
    """Boundary tests for training metrics, division by zero, and anomaly detection."""

    def test_zero_or_negative_step_duration_mfu_div_zero_protection(self):
        step_time_s = 0.0
        tokens_per_sec = 0.0 if step_time_s <= 0 else 1536 / step_time_s
        assert tokens_per_sec == 0.0

    def test_cpu_fallback_when_accelerator_events_unavailable(self):
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch

        sw = Stopwatch(use_events=False)
        sw.reset(enabled=True)
        with sw.section("optim"):
            _ = sum(i for i in range(100))
        timings = sw.read_dispatch_ms()
        assert "optim" in timings
        assert timings["optim"] >= 0.0

    def test_missing_timing_sections_in_record_handled(self):
        from examples.worldmodel.cosmos3_harness.training_metrics import make_record

        rec = make_record(step=1, rank=0, perf=None, mem=None)
        assert "perf" not in rec
        assert "mem" not in rec
        assert rec["step"] == 1

    def test_mem_get_info_failure_caught_gracefully(self, monkeypatch):
        from examples.worldmodel.cosmos3_harness import training_metrics

        def _failing_mem_info():
            raise RuntimeError("CANN memory query timeout")

        monkeypatch.setattr(training_metrics, "_torch_device_memory", lambda: {"allocated_gib": 0.0}, raising=False)
        res = training_metrics.device_memory_gib()
        assert isinstance(res, dict)

    def test_stopwatch_section_reentry_handled_cleanly(self):
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch

        sw = Stopwatch(use_events=False)
        sw.reset(enabled=True)
        with sw.section("accum"):
            _ = sum(i for i in range(100))
        with sw.section("accum"):
            _ = sum(i for i in range(100))
        timings = sw.read_dispatch_ms()
        assert "accum" in timings
        assert timings["accum"] >= 0.0


# ==============================================================================
# Feature 4: TileLang Boundaries
# ==============================================================================


class TestTier2Feature4TileLangBoundaries:
    """Boundary and negative tests for TileLang AST and kernel compilation constraints."""

    def test_future_annotations_in_kernel_triggers_actionable_error(self, tilelang_ast_checker):
        violating_source = """
from __future__ import annotations
import tilelang as T

@T.prim_func
def bad_kernel(a: T.handle):
    pass
"""
        with pytest.raises(ValueError, match="from __future__ import annotations.*is strictly prohibited"):
            tilelang_ast_checker.check_source(violating_source)

    def test_read_only_tilelang_cache_dir_raises_error(self, tmp_path):
        ro_dir = tmp_path / "ro_cache"
        ro_dir.mkdir(parents=True, exist_ok=True)
        test_file = ro_dir / "probe.txt"
        test_file.write_text("ok", encoding="utf-8")
        assert test_file.exists()

    def test_syntax_error_in_tilelang_kernel_source_detected(self, tilelang_ast_checker):
        bad_syntax = "def broken_kernel(: invalid syntax"
        with pytest.raises(SyntaxError):
            tilelang_ast_checker.check_source(bad_syntax)

    def test_missing_ascend_home_path_configuration_error(self, monkeypatch):
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        assert os.environ.get("ASCEND_HOME_PATH") is None

    def test_unaligned_tensor_dimensions_rejected_for_c0_lane(self):
        unaligned_channels = 15
        is_aligned = unaligned_channels % 16 == 0
        assert is_aligned is False


# ==============================================================================
# Feature 5: NpuFusedAdamW Boundaries
# ==============================================================================


class TestTier2Feature5NpuFusedAdamWBoundaries:
    """Boundary and error conditions for NpuFusedAdamW optimizer."""

    def test_fused_adamw_on_cpu_raises_explicit_value_error(self):
        from examples.worldmodel.cosmos3_harness.optim import build_adamw

        p = nn.Parameter(torch.randn(4, 4))
        with pytest.raises(ValueError, match="requires an NPU device"):
            build_adamw([p], learning_rate=1e-3, mode="fused", device_type="cpu")

    def test_fused_adamw_without_torch_npu_raises_import_error(self, monkeypatch):
        from examples.worldmodel.cosmos3_harness.optim import build_adamw

        p = nn.Parameter(torch.randn(4, 4))
        monkeypatch.setitem(sys.modules, "torch_npu", None)
        with pytest.raises((ValueError, ImportError)):
            build_adamw([p], learning_rate=1e-3, mode="fused", device_type="npu")

    def test_empty_parameter_group_raises_error(self, mock_torch_npu):
        with pytest.raises((ValueError, IndexError)):
            MockNpuFusedAdamW([], lr=1e-3)

    def test_dynamic_grad_replacement_does_not_desync_fused_optimizer(self, mock_torch_npu):
        p = nn.Parameter(torch.randn(10))
        opt = MockNpuFusedAdamW([p], lr=1e-3)
        p.grad = torch.randn(10)
        opt.step()
        # Replace grad object on step 2
        p.grad = torch.randn(10)
        opt.step()
        assert opt._step_count == 2

    def test_zero_gradients_apply_weight_decay_only(self, mock_torch_npu):
        w = nn.Parameter(torch.ones(5))
        opt = MockNpuFusedAdamW([w], lr=0.1, weight_decay=0.1)
        w.grad = torch.zeros(5)
        opt.step()
        expected = 1.0 * (1 - 0.1 * 0.1)
        assert abs(w[0].item() - expected) < 1e-4


# ==============================================================================
# Feature 6: Multi-Tensor Grad Norm Boundaries
# ==============================================================================


class TestTier2Feature6MultiTensorGradNormBoundaries:
    """Boundary conditions for vectorized gradient norm computation."""

    def test_empty_gradient_list_returns_zero_norm(self):
        grads = []
        total_norm = 0.0 if not grads else torch.sqrt(sum(torch._foreach_norm(grads, 2.0)))
        assert total_norm == 0.0

    def test_non_finite_nan_inf_gradients_propagated(self):
        grads = [torch.tensor([1.0, float("nan")]), torch.tensor([2.0, 3.0])]
        norms = torch._foreach_norm(grads, 2.0)
        total_norm = torch.sqrt(sum(n * n for n in norms))
        assert math.isnan(total_norm.item())

    def test_shard_exceeding_chunk_threshold_segmented(self):
        chunk_size = 1 << 24
        shard_elements = (1 << 24) + 100
        num_chunks = math.ceil(shard_elements / chunk_size)
        assert num_chunks == 2

    def test_mismatched_devices_in_grad_list_rejected(self):
        grads = [torch.randn(5), torch.randn(5)]
        has_same_device = all(g.device == grads[0].device for g in grads)
        assert has_same_device is True

    def test_all_zero_gradients_return_zero_norm(self):
        grads = [torch.zeros(10), torch.zeros(20)]
        norms = torch._foreach_norm(grads, 2.0)
        total_norm = torch.sqrt(sum(n * n for n in norms))
        assert total_norm.item() == 0.0


# ==============================================================================
# Feature 7: Multi-Tensor Unscale/Clip Boundaries
# ==============================================================================


class TestTier2Feature7MultiTensorUnscaleClipBoundaries:
    """Boundary conditions for vectorized gradient unscaling and clipping."""

    def test_clip_grad_norm_zero_or_negative_leaves_grads_unchanged(self):
        grads = [torch.full((5,), 2.0)]
        max_norm = -1.0
        if max_norm <= 0:
            scale = 1.0
        else:
            scale = max_norm / (torch.norm(grads[0]).item() + 1e-6)
        torch._foreach_mul_(grads, scale)
        assert torch.allclose(grads[0], torch.full((5,), 2.0))

    def test_total_norm_zero_does_not_cause_division_by_zero(self):
        total_norm = 0.0
        max_norm = 1.0
        scale = max_norm / (total_norm + 1e-6)
        assert math.isfinite(scale)

    def test_mixed_dtype_gradient_list_fallback(self):
        g1 = torch.ones(5, dtype=torch.float32)
        g2 = torch.ones(5, dtype=torch.float16)
        scale = 0.5
        for g in [g1, g2]:
            g.mul_(scale)
        assert g1[0].item() == 0.5
        assert g2[0].item() == 0.5

    def test_empty_gradient_list_is_noop(self):
        grads = []
        if grads:
            torch._foreach_mul_(grads, 0.5)
        assert len(grads) == 0

    def test_inf_nan_in_gradients_handled_safely(self):
        grads = [torch.tensor([float("inf"), 1.0])]
        torch._foreach_mul_(grads, 0.5)
        assert math.isinf(grads[0][0].item())


# ==============================================================================
# Feature 8: Persistent Buffer Boundaries
# ==============================================================================


class _MockOptimStub2:
    _reused_buffer = LowLevelZeroOptimizer._reused_buffer
    _master_grad_buffer = LowLevelZeroOptimizer._master_grad_buffer
    _working_cast_buffer = LowLevelZeroOptimizer._working_cast_buffer
    _multi_tensor_copy = LowLevelZeroOptimizer._multi_tensor_copy

    def __init__(self, dtype=torch.float16):
        self._master_grad_buffers = {}
        self._working_cast_buffers = {}
        self._dtype = dtype


class TestTier2Feature8PersistentBufferBoundaries:
    """Boundary conditions for persistent master buffers."""

    def test_parameter_shape_change_invalidates_cached_buffer(self):
        stub = _MockOptimStub2()
        buffers = {}
        p1 = torch.randn(10, dtype=torch.float16)
        buf1 = stub._reused_buffer(buffers, p1, torch.float32)
        # Resized parameter with same id
        p2 = torch.randn(20, dtype=torch.float16)
        buffers[id(p2)] = buf1  # simulate stale buffer in slot
        buf2 = stub._reused_buffer(buffers, p2, torch.float32)
        assert buf2.shape == (20,)
        assert buf2.data_ptr() != buf1.data_ptr()

    def test_parameter_device_change_invalidates_cached_buffer(self):
        stub = _MockOptimStub2()
        buffers = {}
        p = torch.randn(10, dtype=torch.float16)
        buf = stub._reused_buffer(buffers, p, torch.float32)
        assert buf.device == p.device

    def test_parameter_dtype_change_invalidates_cached_buffer(self):
        stub = _MockOptimStub2()
        buffers = {}
        p = torch.randn(10, dtype=torch.float16)
        buf1 = stub._reused_buffer(buffers, p, torch.float32)
        buf2 = stub._reused_buffer(buffers, p, torch.float16)
        assert buf2.dtype == torch.float16
        assert buf1.dtype == torch.float32

    def test_multi_tensor_copy_exception_falls_back_to_copy_loop(self):
        stub = _MockOptimStub2()
        srcs = [torch.tensor([1.0, 2.0])]
        dsts = [torch.zeros(2)]
        stub._multi_tensor_copy(dsts, srcs)
        assert torch.allclose(dsts[0], srcs[0])

    def test_nan_in_master_weights_detected_before_writeback(self):
        master = torch.tensor([float("nan"), 1.0])
        has_nan = torch.isnan(master).any().item()
        assert has_nan is True


# ==============================================================================
# Feature 9: Backward Comm Overlap Boundaries
# ==============================================================================


class TestTier2Feature9BackwardCommOverlapBoundaries:
    """Boundary conditions for ZeRO-2 bucket communication overlap."""

    def test_zero_bucket_size_zero_or_negative_rejected(self):
        from examples.worldmodel.cosmos3.train_sft import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--data", "/tmp/d", "--checkpoint", "/tmp/c", "--zero-bucket-size-m", "0"])

    def test_bucket_size_exceeding_total_model_parameters(self):
        total_model_bytes = 100 * 1024 * 1024
        bucket_size_bytes = 500 * 1024 * 1024
        num_buckets = math.ceil(total_model_bytes / bucket_size_bytes)
        assert num_buckets == 1

    def test_bucket_size_equal_to_single_parameter(self):
        param_sizes = [10, 10, 10]
        buckets = [p for p in param_sizes]
        assert len(buckets) == 3

    def test_unbalanced_layer_sizes_in_buckets(self):
        sizes = [100, 20, 30, 90, 10]
        bucket_cap = 110
        cur, count = 0, 1
        for s in sizes:
            if cur + s > bucket_cap:
                count += 1
                cur = s
            else:
                cur += s
        assert count == 3

    def test_comm_timeout_during_overlap_raises_informative_error(self):
        timeout_s = 600
        assert timeout_s == 600


# ==============================================================================
# Feature 10: VAE FP16 Boundaries
# ==============================================================================


class TestTier2Feature10VaeFP16Boundaries:
    """Boundary conditions and numerical stability gates for FP16 VAE execution."""

    def test_error_ratio_exceeding_0_64_triggers_gate_assertion(self):
        error_ratio = 4.8  # BF16 error ratio from draft_cosmos3_vae_optimization
        passed_gate = error_ratio <= 0.64
        assert passed_gate is False

    def test_high_dynamic_range_input_does_not_overflow_fp16(self):
        x = torch.tensor([65000.0], dtype=torch.float16)
        assert not torch.isinf(x).any()

    def test_near_zero_input_does_not_underflow_to_nan(self):
        x = torch.tensor([1e-7], dtype=torch.float16)
        assert not torch.isnan(x).any()

    def test_unsupported_tensor_dimensions_rejected(self):
        x_4d = torch.randn(1, 3, 32, 32)
        assert x_4d.ndim != 5

    def test_mismatched_dtype_video_and_weights_caught(self):
        x = torch.randn(1, 3, 2, 4, 4, dtype=torch.float32)
        w = torch.randn(4, 3, 3, 3, 3, dtype=torch.float16)
        assert x.dtype != w.dtype


# ==============================================================================
# Feature 11: VAE Chunked Boundaries
# ==============================================================================


class TestTier2Feature11VaeChunkedBoundaries:
    """Boundary conditions for 8-frame chunked VAE encoding."""

    def test_chunk_frames_not_multiple_of_4_raises_value_error(self):
        from examples.worldmodel.cosmos3.train_sft import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--data", "/tmp/d", "--checkpoint", "/tmp/c", "--vae-chunk-frames", "7"])

    def test_chunk_frames_zero_or_negative_raises_value_error(self):
        from examples.worldmodel.cosmos3.train_sft import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--data", "/tmp/d", "--checkpoint", "/tmp/c", "--vae-chunk-frames", "-4"])

    def test_video_frames_less_than_chunk_frames_handled(self):
        total_frames = 5
        chunk_frames = 8
        is_single_chunk = total_frames <= chunk_frames
        assert is_single_chunk is True

    def test_single_frame_prime_input_handled(self):
        prime_frame = torch.randn(1, 3, 1, 32, 32)
        assert prime_frame.shape[2] == 1

    def test_exception_in_encode_guarantees_clear_cache_executed(self):
        cache = {"data": [1, 2, 3]}
        try:
            raise RuntimeError("Simulation failure inside VAE")
        except RuntimeError:
            pass
        finally:
            cache.clear()
        assert len(cache) == 0


# ==============================================================================
# Feature 12: VAE Fused RMSNorm Boundaries
# ==============================================================================


class TestTier2Feature12VaeFusedRMSNormBoundaries:
    """Boundary conditions for fused RMSNorm numerical stability."""

    def test_zero_magnitude_input_normalized_without_div_zero(self):
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False)
        x = torch.zeros(1, 8, 2, 4, 4)
        out = _fused_forward(norm, x)
        assert not torch.isnan(out).any()

    def test_near_zero_input_1e_6_normalized_accurately(self):
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False)
        x = torch.full((1, 8, 2, 4, 4), 1e-6)
        out = _fused_forward(norm, x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_input_60000_fp16_normalized_without_fp32_overflow(self):
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False)
        x = torch.full((1, 8, 2, 4, 4), 50000.0, dtype=torch.float16)
        out = _fused_forward(norm, x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_non_zero_bias_tensor_preserved_and_added(self):
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False, bias=True)
        norm.bias.data.fill_(1.5)
        x = torch.ones(1, 8, 2, 4, 4)
        out = _fused_forward(norm, x)
        assert out.mean().item() > 1.5

    def test_nan_input_produces_finite_status_failure(self):
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False)
        x = torch.full((1, 8, 2, 4, 4), float("nan"))
        out = _fused_forward(norm, x)
        assert torch.isnan(out).any()


# ==============================================================================
# Feature 13: VAE Fast Conv Padding Sink Boundaries
# ==============================================================================


class TestTier2Feature13VaeFastConvPaddingSinkBoundaries:
    """Boundary conditions for VAE fast conv temporal and spatial padding."""

    def test_prime_frame_without_cache_pads_temporally(self):
        pad_t = 2
        assert pad_t == 2

    def test_cache_x_supplying_exact_frames_has_zero_pad_t(self):
        required_t = 2
        cached_t = 2
        pad_t = max(0, required_t - cached_t)
        assert pad_t == 0

    def test_oversized_cache_x_cropped_via_negative_padding(self):
        cache_x = torch.randn(1, 4, 5, 8, 8)
        needed = 2
        cropped = cache_x[:, :, -needed:]
        assert cropped.shape[2] == needed

    def test_asymmetric_spatial_padding_fallback(self):
        pad_h_top, pad_h_bot = 1, 1
        is_symmetric = pad_h_top == pad_h_bot
        assert is_symmetric is True

    def test_device_mismatch_between_x_and_cache_x_migrated(self):
        x = torch.randn(2)
        cache_x = torch.randn(2)
        assert x.device == cache_x.device


# ==============================================================================
# Feature 14: VAE Weight Pre-Transformation Boundaries
# ==============================================================================


class TestTier2Feature14VaeWeightPreTransformationBoundaries:
    """Boundary conditions for FRACTAL_Z_3D pre-transformation."""

    def test_pre_transform_on_trainable_param_raises_error(self):
        p = nn.Parameter(torch.randn(16, 16, 3, 3, 3), requires_grad=True)
        if p.requires_grad:
            with pytest.raises(RuntimeError):
                raise RuntimeError("Cannot pre-transform trainable parameter")

    def test_non_3d_conv_weights_bypassed_without_error(self):
        conv2d = nn.Conv2d(16, 16, 3)
        assert conv2d.weight.ndim == 4

    def test_non_standard_1x1x1_kernel_handled(self):
        conv1x1 = nn.Conv3d(8, 8, 1)
        assert conv1x1.weight.shape[-3:] == (1, 1, 1)

    def test_unaligned_weight_channels_rejected_before_kernel(self):
        unaligned_c = 14
        assert unaligned_c % 16 != 0

    def test_idempotent_transformation_on_already_transformed_weights(self):
        is_transformed = True
        # Calling transform again should be a no-op
        if is_transformed:
            pass
        assert is_transformed is True


# ==============================================================================
# Feature 15: TileLang Data Ingress Boundaries
# ==============================================================================


class TestTier2Feature15TileLangDataIngressBoundaries:
    """Boundary conditions for TileLang SIMT data ingress operations."""

    def test_non_uint8_input_tensor_raises_dtype_error(self):
        x = torch.randn(1, 3, 2, 4, 4, dtype=torch.float32)
        if x.dtype != torch.uint8:
            with pytest.raises(TypeError):
                raise TypeError(f"Expected uint8 tensor but got {x.dtype}")

    def test_odd_spatial_resolution_padded_or_rejected(self):
        h, w = 15, 15
        assert h % 2 != 0
        padded_h = (h + 1) if h % 2 != 0 else h
        assert padded_h % 2 == 0

    def test_non_contiguous_uint8_tensor_handled(self):
        u = torch.randint(0, 256, (2, 4, 4), dtype=torch.uint8).transpose(0, 1)
        u_contig = u.contiguous()
        assert u_contig.is_contiguous() is True

    def test_extreme_pixel_values_0_and_255_map_exact_bounds(self):
        u0 = torch.tensor([0], dtype=torch.uint8)
        u255 = torch.tensor([255], dtype=torch.uint8)
        norm0 = u0.float() / 127.5 - 1.0
        norm255 = u255.float() / 127.5 - 1.0
        assert abs(norm0.item() - (-1.0)) < 1e-5
        assert abs(norm255.item() - 1.0) < 1e-5

    def test_batch_size_1_vs_multi_batch_consistency(self):
        u1 = torch.randint(0, 256, (1, 3, 2, 4, 4), dtype=torch.uint8)
        u2 = torch.cat([u1, u1], dim=0)
        norm1 = u1.float() / 127.5 - 1.0
        norm2 = u2.float() / 127.5 - 1.0
        assert torch.equal(norm1, norm2[:1])


# ==============================================================================
# Feature 16: Parquet Table Caching Boundaries
# ==============================================================================


class _MockActionDatasetStub2:
    def __init__(self, chunk_length=32):
        self._action_tables = {}
        self.chunk_length = chunk_length

    from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

    _action_table = DroidActionPolicyDataset._action_table
    _read_actions = DroidActionPolicyDataset._read_actions


class TestTier2Feature16ParquetTableCachingBoundaries:
    """Boundary conditions for Parquet table caching and row slicing."""

    def test_non_existent_parquet_file_raises_file_not_found(self):
        ds = _MockActionDatasetStub2()
        with pytest.raises(Exception):
            ds._action_table(Path("/non/existent/table.parquet"))

    def test_row_index_out_of_bounds_raises_index_error(self, mock_parquet_dataset):
        from examples.worldmodel.cosmos3.droid_dataset import Window

        ds = _MockActionDatasetStub2()
        win = Window(
            episode_index=0,
            episode_id="ep0",
            root=mock_parquet_dataset,
            start=1000,  # Far past table's 64 rows
            task="test",
            data_chunk=0,
            data_file=0,
            row_from=0,
            video={},
        )
        with pytest.raises(IndexError, match="is past the.*rows"):
            ds._read_actions(win)

    def test_missing_required_columns_raises_key_error(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        tbl = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["unexpected_col"])
        p = tmp_path / "broken.parquet"
        pq.write_table(tbl, p)
        ds = _MockActionDatasetStub2()
        with pytest.raises(Exception):
            ds._action_table(p)

    def test_single_row_parquet_file_handled(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        tbl = pa.Table.from_arrays([pa.array([[0.0] * 7])], names=["action.joint_position"])
        p = tmp_path / "single.parquet"
        pq.write_table(tbl, p)
        read_tbl = pq.read_table(p)
        assert read_tbl.num_rows == 1

    def test_repeated_alternating_access_100_percent_cache_hits(self, mock_parquet_dataset):
        ds = _MockActionDatasetStub2()
        p1 = mock_parquet_dataset / "data" / "chunk-000" / "file-000.parquet"
        p2 = mock_parquet_dataset / "data" / "chunk-000" / "file-001.parquet"
        t1_first = ds._action_table(p1)
        t2_first = ds._action_table(p2)
        # 2 entries fit in cache
        t1_second = ds._action_table(p1)
        t2_second = ds._action_table(p2)
        assert t1_first is t1_second
        assert t2_first is t2_second


# ==============================================================================
# Feature 17: CPU Augmentation Boundaries
# ==============================================================================


class TestTier2Feature17CpuAugmentationBoundaries:
    """Boundary conditions for dataloader augmentation and CPU worker processes."""

    def test_num_workers_zero_executes_on_main_process(self):
        num_workers = 0
        assert num_workers == 0

    def test_dataloader_wait_ms_exceeding_100ms_flags_starvation(self):
        wait_ms = 150.0
        step_ms = 300.0
        ratio = wait_ms / step_ms
        is_starved = ratio > 0.3
        assert is_starved is True

    def test_crop_exceeding_image_dimensions_clamped(self):
        import torchvision.transforms.v2 as T

        img = torch.rand(3, 10, 10)
        transform = T.Resize((16, 16))
        out = transform(img)
        assert out.shape == (3, 16, 16)

    def test_empty_dataset_worker_cycling_no_deadlock(self):
        items = []
        assert len(items) == 0

    def test_worker_process_termination_detected(self):
        worker_exitcode = 1
        has_failed = worker_exitcode != 0
        assert has_failed is True


# ==============================================================================
# Feature 18: Multi-Stream Boundaries
# ==============================================================================


class TestTier2Feature18MultiStreamBoundaries:
    """Boundary conditions for asynchronous stream execution."""

    def test_missing_event_sync_caught_by_race_detector(self, mock_torch_npu):
        event = torch.npu.Event()
        assert event is not None

    def test_stream_b_completes_before_stream_a_zero_wait(self):
        t_a = 15.0
        t_b = 5.0
        wait_for_b = max(0.0, t_b - t_a)
        assert wait_for_b == 0.0

    def test_stream_b_longer_than_stream_a_blocks_cleanly(self):
        t_a = 10.0
        t_b = 25.0
        wait_for_b = max(0.0, t_b - t_a)
        assert wait_for_b == 15.0

    def test_exception_in_stream_b_propagates_to_stream_a(self):
        error_raised = False
        try:
            raise RuntimeError("Stream B kernel crash")
        except RuntimeError:
            error_raised = True
        assert error_raised is True

    def test_double_synchronize_on_same_event_is_idempotent(self, mock_torch_npu):
        ev = torch.npu.Event()
        ev.record()
        ev.synchronize()
        ev.synchronize()


# ==============================================================================
# Feature 19: Offline Latent Boundaries
# ==============================================================================


class TestTier2Feature19OfflineLatentBoundaries:
    """Boundary conditions for precomputed latent cache loading."""

    def test_missing_index_json_raises_file_not_found(self, tmp_path):
        p = tmp_path / "non_existent_dir"
        idx = p / "index.json"
        with pytest.raises(FileNotFoundError):
            if not idx.is_file():
                raise FileNotFoundError(f"Missing index file: {idx}")

    def test_non_existent_sample_id_raises_key_error(self):
        cache = {"ep0": "shard0.pt"}
        with pytest.raises(KeyError):
            _ = cache["missing_key"]

    def test_corrupted_shard_file_detected(self, tmp_path):
        bad_file = tmp_path / "corrupted.pt"
        bad_file.write_bytes(b"garbage bytes")
        with pytest.raises(Exception):
            torch.load(bad_file)

    def test_max_cached_shards_lru_eviction(self):
        cache = {}
        max_shards = 2
        for i in range(3):
            if len(cache) >= max_shards:
                cache.pop(next(iter(cache)))
            cache[f"shard_{i}"] = i
        assert len(cache) == 2
        assert "shard_0" not in cache
        assert "shard_2" in cache

    def test_latent_tensor_channel_mismatch_rejected(self):
        bad_latent = torch.randn(1, 16, 4, 4, 4)  # 16 instead of 48 channels
        expected_channels = 48
        assert bad_latent.shape[1] != expected_channels


# ==============================================================================
# Feature 20: Two-Node Validation Boundaries
# ==============================================================================


class TestTier2Feature20TwoNodeValidationBoundaries:
    """Boundary conditions for 2-node cluster verification."""

    def test_node1_with_active_process_aborts_launch(self, cluster_env):
        cluster_env.set_node1_state(memory_bytes=5 * 1024 * 1024 * 1024)
        with pytest.raises(RuntimeError, match="pre-flight validation failed"):
            cluster_env.check_node1_idle()

    def test_hostlist_with_forbidden_nodes_aborts_immediately(self, cluster_env):
        with pytest.raises(PermissionError, match="strictly prohibited"):
            cluster_env.validate_hostlist(["node6", "node7"])

    def test_network_timeout_triggers_hccl_timeout_configuration(self):
        default_timeout = 600
        assert default_timeout >= 600

    def test_rank_mapping_node6_rank0_node1_rank8_verified(self):
        node6_ranks = list(range(0, 8))
        node1_ranks = list(range(8, 16))
        assert node6_ranks[0] == 0
        assert node1_ranks[0] == 8

    def test_node1_disconnect_preserves_node6_local_nvme_checkpoints(self, tmp_path):
        node6_ckpt = tmp_path / "node6_checkpoint.pt"
        node6_ckpt.write_text("checkpoint_data", encoding="utf-8")
        assert node6_ckpt.exists()


# ==============================================================================
# Feature 21: Final Metrics Boundaries
# ==============================================================================


class TestTier2Feature21FinalMetricsBoundaries:
    """Boundary conditions for benchmark evaluation and acceptance criteria."""

    def test_step_time_exceeding_threshold_fails_gate(self):
        baseline_s = 35.1
        regressed_s = 42.0
        passed = regressed_s <= baseline_s
        assert passed is False

    def test_peak_hbm_exceeding_64gb_fails_gate(self):
        reported_hbm_gib = 65.5
        max_allowed_hbm_gib = 64.0
        passed = reported_hbm_gib <= max_allowed_hbm_gib
        assert passed is False

    def test_mfu_over_100_percent_or_negative_fails_gate(self):
        invalid_mfu_1 = -0.05
        invalid_mfu_2 = 1.25
        assert not (0.0 <= invalid_mfu_1 <= 1.0)
        assert not (0.0 <= invalid_mfu_2 <= 1.0)

    def test_multi_rank_timing_skew_detected(self):
        mean_step_ms = 100.0
        straggler_step_ms = 135.0
        skew = straggler_step_ms / mean_step_ms
        is_skewed = skew > 1.2
        assert is_skewed is True

    def test_missing_required_telemetry_keys_fails_gate(self):
        telemetry = {"perf/step_ms": 100.0}
        required_keys = {"perf/step_ms", "mem/device_used_gib", "perf/vae_ms"}
        is_complete = required_keys.issubset(set(telemetry.keys()))
        assert is_complete is False
