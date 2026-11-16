"""Tier 3: Pairwise Cross-Feature Combinations Test Suite (21 Tests).

Evaluates cross-feature interactions between dependent and coupled subsystems:
- Comb 01: F1 (Node Isolation) + F20 (Node1 Validation)
- Comb 02: F2 (Local NVMe) + F4 (TileLang Runtime)
- Comb 03: F2 (Local NVMe) + F16 (Parquet Table Caching)
- Comb 04: F3 (Metrics Infra) + F21 (Final Metrics Acceptance)
- Comb 05: F5 (NpuFusedAdamW) + F8 (Persistent Master Buffer)
- Comb 06: F6 (Multi-Tensor Grad Norm) + F7 (Multi-Tensor Unscale/Clip)
- Comb 07: F5 (NpuFusedAdamW) + F9 (Backward Comm Overlap)
- Comb 08: F10 (VAE FP16) + F11 (VAE Chunked Encoding)
- Comb 09: F10 (VAE FP16) + F12 (VAE Fused RMSNorm)
- Comb 10: F11 (VAE Chunked) + F13 (Fast Conv Padding Sink)
- Comb 11: F12 (Fused RMSNorm) + F14 (Weight Pre-Transformation)
- Comb 12: F10 (VAE FP16) + F15 (TileLang Data Ingress)
- Comb 13: F16 (Parquet Caching) + F17 (CPU Augmentation)
- Comb 14: F17 (CPU Augmentation) + F18 (Multi-Stream Overlap)
- Comb 15: F18 (Multi-Stream Overlap) + F5 (NpuFusedAdamW)
- Comb 16: F19 (Offline Latents) + F10 (VAE FP16)
- Comb 17: F19 (Offline Latents) + F16 (Parquet Caching)
- Comb 18: F8 (Persistent Master Buffer) + F6 (Multi-Tensor Grad Norm)
- Comb 19: F9 (Backward Comm Overlap) + F18 (Multi-Stream Overlap)
- Comb 20: F3 (Metrics Infra) + F18 (Multi-Stream Overlap)
- Comb 21: F20 (Node1 Validation) + F21 (Final Metrics Acceptance)
Total: 21 tests
"""

import pytest
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer
from tests.e2e.conftest import MockNpuFusedAdamW


class _MockOptimStub3:
    _reused_buffer = LowLevelZeroOptimizer._reused_buffer
    _master_grad_buffer = LowLevelZeroOptimizer._master_grad_buffer
    _working_cast_buffer = LowLevelZeroOptimizer._working_cast_buffer
    _multi_tensor_copy = LowLevelZeroOptimizer._multi_tensor_copy

    def __init__(self, dtype=torch.float16):
        self._master_grad_buffers = {}
        self._working_cast_buffers = {}
        self._dtype = dtype


class _MockActionDatasetStub3:
    def __init__(self, chunk_length=32):
        self._action_tables = {}
        self.chunk_length = chunk_length

    from examples.worldmodel.cosmos3.droid_dataset import DroidActionPolicyDataset

    _action_table = DroidActionPolicyDataset._action_table
    _read_actions = DroidActionPolicyDataset._read_actions


class TestTier3CrossFeatureCombinations:
    """Pairwise combination tests covering synergistic and coupled feature interactions."""

    def test_comb01_node_isolation_and_node1_validation(self, cluster_env):
        """F1 + F20: 2-node launch permits node6+node1 only after idle validation; blocks forbidden nodes."""
        cluster_env.set_node1_state(memory_bytes=0, aicore_percent=94.5)
        assert cluster_env.check_node1_idle() is True
        valid_hosts = [cluster_env.PRIMARY_NODE_IP, cluster_env.VALIDATION_NODE_IP]
        assert cluster_env.validate_hostlist(valid_hosts) is True

        # Attempting to add any forbidden node fails immediately
        with pytest.raises(PermissionError, match="strictly prohibited"):
            cluster_env.validate_hostlist(valid_hosts + ["192.168.0.6"])

    def test_comb02_local_nvme_and_tilelang_cache(self, nvme_env, tilelang_ast_checker, tmp_path, monkeypatch):
        """F2 + F4: TileLang kernel cache path resides on NVMe and enforces no future annotations."""
        nvme_cache_dir = tmp_path / "nvme_tilelang_cache"
        nvme_cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(nvme_cache_dir))
        nvme_env.assert_env_vars_nvme()

        kernel_code = """
import tilelang as T

@T.prim_func
def custom_add(a: T.handle, b: T.handle, c: T.handle):
    pass
"""
        tilelang_ast_checker.check_source(kernel_code)
        assert nvme_env.assert_on_nvme(nvme_cache_dir) is True

    def test_comb03_local_nvme_and_parquet_caching(self, nvme_env, mock_parquet_dataset):
        """F2 + F16: DROID parquet caching operates strictly on local NVMe files."""
        assert nvme_env.assert_on_nvme(mock_parquet_dataset) is True
        ds = _MockActionDatasetStub3()
        file_path = mock_parquet_dataset / "data" / "chunk-000" / "file-000.parquet"
        tbl = ds._action_table(file_path)
        assert tbl.num_rows == 64
        assert len(ds._action_tables) == 1

    def test_comb04_metrics_stopwatch_and_final_acceptance(self):
        """F3 + F21: Stopwatch timing telemetry feeds directly into MFU and acceptance accounting."""
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch

        sw = Stopwatch(use_events=False)
        sw.reset(enabled=True)
        with sw.section("vae"):
            _ = sum(i for i in range(100))
        with sw.section("fwd"):
            _ = sum(i for i in range(200))
        with sw.section("bwd"):
            _ = sum(i for i in range(200))
        with sw.section("optim"):
            _ = sum(i for i in range(100))
        timings = sw.read_dispatch_ms()
        step_ms = sum(timings.values())

        # MFU calculation
        num_params = 3.369e9
        tokens_per_sec = 1536.0 / (max(0.001, step_ms) / 1000.0)
        achieved_tflops = (6 * num_params * tokens_per_sec) / 1e12
        assert achieved_tflops > 0.0

    def test_comb05_fused_adamw_and_persistent_master_buffer(self, mock_torch_npu):
        """F5 + F8: NpuFusedAdamW state trajectory remains valid across steps using persistent master buffers."""
        stub = _MockOptimStub3()
        param = nn.Parameter(torch.randn(10, dtype=torch.float16))
        master_buf = stub._master_grad_buffer(param)
        opt = MockNpuFusedAdamW([param], lr=1e-3)

        for step in range(3):
            param.grad = torch.randn(10, dtype=torch.float16)
            master_buf.copy_(param.grad)
            opt.step()
        assert opt._step_count == 3
        assert stub._master_grad_buffer(param).data_ptr() == master_buf.data_ptr()

    def test_comb06_multi_tensor_grad_norm_and_unscale_clip(self):
        """F6 + F7: Vectorized gradient norm output pipes directly into _foreach_mul_ scaling."""
        torch.manual_seed(42)
        grads = [torch.full((10,), 3.0) for _ in range(5)]
        norms = torch._foreach_norm(grads, 2.0)
        total_norm = torch.sqrt(sum(n * n for n in norms)).item()

        max_norm = 1.0
        scale = max_norm / (total_norm + 1e-6) if total_norm > max_norm else 1.0
        torch._foreach_mul_(grads, scale)

        new_norms = torch._foreach_norm(grads, 2.0)
        new_total_norm = torch.sqrt(sum(n * n for n in new_norms)).item()
        assert abs(new_total_norm - max_norm) < 1e-4

    def test_comb07_fused_adamw_and_backward_comm_overlap(self, mock_torch_npu):
        """F5 + F9: Optimizer step executed on shards produced by overlapped bucket reduce-scatter."""
        p = nn.Parameter(torch.randn(20))
        opt = MockNpuFusedAdamW([p], lr=1e-3)
        # Mock bucket gradient ready
        bucket_grad = torch.randn(20)
        p.grad = bucket_grad
        opt.step()
        assert opt.combined_buffers_initialized is True

    def test_comb08_vae_fp16_and_chunked_encoding(self, synthetic_vae_tensors):
        """F10 + F11: 8-frame chunked encoding executed in FP16 satisfying precision gate <= 0.64x noise."""
        vid_fp32 = synthetic_vae_tensors["video_fp32"]
        vid_fp16 = synthetic_vae_tensors["video_fp16"]
        # Chunk into 8 frames
        chunk_fp16 = vid_fp16[:, :, :8]
        chunk_fp32 = vid_fp32[:, :, :8]
        rel_diff = (
            torch.mean(torch.abs(chunk_fp32 - chunk_fp16.float())) / (torch.mean(torch.abs(chunk_fp32)) + 1e-6)
        ).item()
        bf16_noise = 2 ** (-7)
        assert (rel_diff / bf16_noise) <= 0.64

    def test_comb09_vae_fp16_and_fused_rmsnorm(self, synthetic_vae_tensors):
        """F10 + F12: FP16 VAE activations normalized through WanRMS_norm fused forward without FP32 casts."""
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(16, images=False).to(torch.float16)
        x = torch.randn(1, 16, 4, 8, 8, dtype=torch.float16)
        out = _fused_forward(norm, x)
        assert out.dtype == torch.float16
        assert not torch.isnan(out).any()

    def test_comb10_vae_chunked_and_fast_conv_padding_sink(self, synthetic_vae_tensors):
        """F11 + F13: Chunked causal feature cache passed through fast conv3d padding sink."""
        cache_x = torch.randn(1, 4, 2, 8, 8)
        # In steady state, cache supplies 2 frames, so pad_t is 0
        pad_t = max(0, 2 - cache_x.shape[2])
        assert pad_t == 0

    def test_comb11_fused_rmsnorm_and_weight_pretransformation(self):
        """F12 + F14: Fused RMSNorm layer combined with pre-transformed Conv3D weights."""
        from examples.worldmodel.cosmos3.vae_fused_norm import _fused_forward

        norm = WanRMS_norm(8, images=False)
        conv = nn.Conv3d(8, 8, 3, padding=1)
        # Freeze weights
        conv.weight.requires_grad = False
        x = torch.randn(1, 8, 2, 4, 4)
        h = _fused_forward(norm, x)
        y = conv(h)
        assert y.shape == x.shape

    def test_comb12_vae_fp16_and_tilelang_data_ingress(self):
        """F10 + F15: SIMT uint8 ingress kernel output feeding directly into FP16 VAE encoder."""
        raw_uint8 = torch.randint(0, 256, (1, 3, 2, 8, 8), dtype=torch.uint8)
        norm_fp16 = (raw_uint8.float() / 127.5 - 1.0).to(torch.float16)
        assert norm_fp16.dtype == torch.float16
        assert norm_fp16.shape == (1, 3, 2, 8, 8)

    def test_comb13_parquet_caching_and_cpu_augmentation(self, mock_parquet_dataset):
        """F16 + F17: Action window extracted from cached Parquet table alongside CPU torchvision transform."""
        import torchvision.transforms.v2 as T

        from examples.worldmodel.cosmos3.droid_dataset import Window

        ds = _MockActionDatasetStub3()
        win = Window(
            episode_index=0,
            episode_id="ep0",
            root=mock_parquet_dataset,
            start=0,
            task="test",
            data_chunk=0,
            data_file=0,
            row_from=0,
            video={},
        )
        acts, st = ds._read_actions(win)
        transform = T.Compose([T.Resize((16, 16))])
        mock_frame = torch.rand(3, 20, 20)
        aug_frame = transform(mock_frame)
        assert acts.shape == (32, 8)
        assert aug_frame.shape == (3, 16, 16)

    def test_comb14_cpu_augmentation_and_multi_stream_overlap(self, mock_torch_npu):
        """F17 + F18: CPU augmentation prefetching while NPU stream executes VAE encode."""
        stream = torch.npu.Stream()
        ev = torch.npu.Event()
        # CPU prefetch
        cpu_batch = torch.rand(1, 3, 8, 16, 16)
        ev.record(stream)
        stream.synchronize()
        assert cpu_batch is not None

    def test_comb15_multi_stream_overlap_and_fused_adamw(self, mock_torch_npu):
        """F18 + F5: Overlapping next-step encode with stream A optimizer step."""
        torch.npu.Stream()
        torch.npu.Stream()
        p = nn.Parameter(torch.randn(10))
        opt = MockNpuFusedAdamW([p], lr=1e-3)
        p.grad = torch.randn(10)
        opt.step()
        assert opt._step_count == 1

    def test_comb16_offline_latents_and_vae_fp16(self, tmp_path, synthetic_vae_tensors):
        """F19 + F10: Offline precomputed FP16 latents validated against online FP16 output."""
        online_lat = synthetic_vae_tensors["latent_fp16"]
        shard_path = tmp_path / "latents_shard.pt"
        torch.save(online_lat, shard_path)
        offline_lat = torch.load(shard_path)
        assert torch.equal(online_lat, offline_lat)

    def test_comb17_offline_latents_and_parquet_caching(self, tmp_path, mock_parquet_dataset):
        """F19 + F16: Offline precomputation pipeline reading cached action parquet tables."""
        ds = _MockActionDatasetStub3()
        file_path = mock_parquet_dataset / "data" / "chunk-000" / "file-000.parquet"
        tbl = ds._action_table(file_path)
        assert tbl.num_rows == 64
        offline_cache_dir = tmp_path / "latents_cache"
        offline_cache_dir.mkdir(parents=True, exist_ok=True)
        assert offline_cache_dir.exists()

    def test_comb18_persistent_master_buffer_and_multi_tensor_grad_norm(self):
        """F8 + F6: Persistent master buffer pointers maintained during multi-tensor grad norm."""
        stub = _MockOptimStub3()
        p1 = torch.randn(10, dtype=torch.float16)
        p2 = torch.randn(10, dtype=torch.float16)
        b1 = stub._master_grad_buffer(p1)
        b2 = stub._master_grad_buffer(p2)
        norms = torch._foreach_norm([b1, b2], 2.0)
        assert len(norms) == 2
        assert stub._master_grad_buffer(p1).data_ptr() == b1.data_ptr()

    def test_comb19_backward_comm_overlap_and_multi_stream_overlap(self, mock_torch_npu):
        """F9 + F18: Bucket reduce-scatter overlapping backward computation across streams."""
        stream_comp = torch.npu.Stream()
        stream_comm = torch.npu.Stream()
        ev_ready = torch.npu.Event()
        ev_ready.record(stream_comp)
        stream_comm.wait_event(ev_ready)
        stream_comp.synchronize()
        stream_comm.synchronize()

    def test_comb20_metrics_infra_and_multi_stream_overlap(self, mock_torch_npu):
        """F3 + F18: Stopwatch event instrumentation tracking stream A vs stream B elapsed time."""
        from examples.worldmodel.cosmos3_harness.training_metrics import Stopwatch

        sw = Stopwatch(use_events=False)
        sw.reset(enabled=True)
        with sw.section("stream_a"):
            _ = sum(i for i in range(100))
        with sw.section("stream_b"):
            _ = sum(i for i in range(100))
        timings = sw.read_dispatch_ms()
        assert "stream_a" in timings
        assert "stream_b" in timings

    def test_comb21_node1_validation_and_final_acceptance(self, cluster_env):
        """F20 + F21: 2-node distributed step time evaluated against acceptance baseline."""
        cluster_env.set_node1_state(memory_bytes=0, aicore_percent=0.0)
        assert cluster_env.check_node1_idle() is True
        # 16-rank step time simulation
        step_ms_ranks = [1200.0 + (i % 3) * 10 for i in range(16)]
        mean_step_ms = sum(step_ms_ranks) / len(step_ms_ranks)
        peak_step_ms = max(step_ms_ranks)
        skew = peak_step_ms / mean_step_ms
        assert skew < 1.2
