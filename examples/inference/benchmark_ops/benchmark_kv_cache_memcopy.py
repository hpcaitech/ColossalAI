import torch

from colossalai.inference.modeling.layers.attention import copy_to_cache
from colossalai.kernel.kernel_loader import InferenceOpsLoader
from colossalai.kernel.triton import copy_kv_to_blocked_cache
from colossalai.utils import get_current_device
from tests.test_infer.test_kernels.cuda.test_kv_cache_memcpy import prepare_data as prepare_data_new_kcache_layout
from tests.test_infer.test_kernels.triton.test_kvcache_copy import prepare_data

try:
    import triton  # noqa
except ImportError:
    print("please install triton from https://github.com/openai/triton")

inference_ops = InferenceOpsLoader().load()

HEAD_DIM = 128
BATCH = 16
BLOCK_SIZE = 32
SAME_LEN = True
WARM_UPS = 10
REPS = 100
configs = [
    triton.testing.Benchmark(
        x_names=["KV_SEQ_LEN"],
        x_vals=[2**i for i in range(8, 13)],
        line_arg="provider",
        line_vals=["torch_copy_func", "triton_copy_func", "triton_new_kcache_layout", "cuda_copy_func"],
        line_names=["torch_copy_func", "triton_copy_func", "triton_new_kcache_layout", "cuda_copy_func"],
        styles=[("red", "-"), ("blue", "-"), ("yellow", "-"), ("green", "-")],
        ylabel="ms",
        plot_name=f"kvcache_copy_decoding_stage-batch-{BATCH}",
        args={"bsz": BATCH, "block_size": 16, "max_seq_len": 8192, "num_kv_heads": 16, "same_context_len": True},
    )
]


@triton.testing.perf_report(configs)
def benchmark_kvcache_copy(
    provider: str,
    bsz: int,
    block_size: int,
    max_seq_len: int,
    KV_SEQ_LEN: int,  # maximum past kv length (unequal context lens in batch) or past kv len (equal context lens)
    num_kv_heads: int,
    same_context_len: bool,
):
    dtype = torch.float16
    device = get_current_device()

    assert KV_SEQ_LEN <= max_seq_len, "Assigned maximum kv length must be smaller or equal to maximum seq len"

    new_k, new_v, k_cache, v_cache, context_lengths, block_tables = prepare_data(
        bsz,
        num_kv_heads,
        HEAD_DIM,
        block_size,
        max_seq_len // block_size,
        same_context_len,
        KV_SEQ_LEN,
        device=device,
        dtype=dtype,
    )

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch_copy_func":
        fn = lambda: copy_to_cache(new_k, k_cache, lengths=context_lengths, block_tables=block_tables, type="decoding")
    elif provider == "triton_copy_func":
        fn = lambda: copy_kv_to_blocked_cache(new_k, new_v, k_cache, v_cache, context_lengths, block_tables)
    elif provider == "triton_new_kcache_layout":
        # NOTE New kcache layout (num_blocks, num_kv_heads, head_dim // x, block_size, x) to be applied
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        k_cache_shape = (bsz * max_seq_len // block_size, num_kv_heads, HEAD_DIM // x, block_size, x)
        k_cache = torch.zeros(size=k_cache_shape, dtype=dtype, device=device)  # update k_cache layout
        fn = lambda: copy_kv_to_blocked_cache(
            new_k, new_v, k_cache, v_cache, context_lengths, block_tables, use_new_kcache_layout=True
        )
    elif provider == "cuda_copy_func":
        _, _, k_cache, _, _, _, _, _, _ = prepare_data_new_kcache_layout(
            bsz, num_kv_heads, block_size, max_seq_len // block_size, context_lengths - 1, device, dtype
        )
        new_k = new_k.squeeze(1) if new_k.dim() == 4 else new_k
        new_v = new_v.squeeze(1) if new_v.dim() == 4 else new_v
        fn = lambda: inference_ops.decode_kv_cache_memcpy(new_k, new_v, k_cache, v_cache, context_lengths, block_tables)

    ms, min_ms, max_ms = triton.testing.do_bench(fn, warmup=WARM_UPS, rep=REPS, quantiles=quantiles)
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark_kvcache_copy.run(save_path=".", print_data=True)
