import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("please install triton from https://github.com/openai/triton")

if HAS_TRITON:
    # adapted from https://github.com/ModelTC/lightllm/blob/5c559dd7981ed67679a08a1e09a88fb4c1550b3a/lightllm/common/triton_kernel/destindex_copy_kv.py
    @triton.jit
    def _fwd_copy_kv_cache_dest(
        kv_cache_ptr,
        dest_index_ptr,
        out,
        stride_k_bs,
        stride_k_h,
        stride_k_d,
        stride_o_bs,
        stride_o_h,
        stride_o_d,
        head_num,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_HEAD: tl.constexpr,
    ):
        cur_index = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_HEAD)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        dest_index = tl.load(dest_index_ptr + cur_index)

        cache_offsets = stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
        k_ptrs = kv_cache_ptr + cur_index * stride_k_bs + cache_offsets

        o_offsets = stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
        o_ptrs = out + dest_index * stride_o_bs + o_offsets

        k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
        tl.store(o_ptrs, k, mask=offs_h[:, None] < head_num)
        return

    # adepted from https://github.com/ModelTC/lightllm/blob/5c559dd7981ed67679a08a1e09a88fb4c1550b3a/lightllm/common/triton_kernel/destindex_copy_kv.py
    @torch.no_grad()
    def copy_kv_cache_to_dest(k_ptr, dest_index_ptr, out):
        seq_len = dest_index_ptr.shape[0]
        head_num = k_ptr.shape[1]
        head_dim = k_ptr.shape[2]
        assert head_num == out.shape[1], "head_num should be the same for k_ptr and out"
        assert head_dim == out.shape[2], "head_dim should be the same for k_ptr and out"

        num_warps = 2
        _fwd_copy_kv_cache_dest[(seq_len,)](
            k_ptr,
            dest_index_ptr,
            out,
            k_ptr.stride(0),
            k_ptr.stride(1),
            k_ptr.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            head_num,
            BLOCK_DMODEL=head_dim,
            BLOCK_HEAD=triton.next_power_of_2(head_num),
            num_warps=num_warps,
            num_stages=2,
        )
        return
