import torch
import triton
import triton.language as tl


@triton.jit
def prefill_cache_kernel(
    CaChe,
    cumsum_lengths,
    output,
    cache_stride,
    hidden_stride,
    total_length,
    HIDDEN_DIM: tl.constexpr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx0 = tl.program_id(axis=0)
    idx1 = tl.program_id(axis=1)
    idx = idx0 * BLOCK_SIZE + idx1

    # original seq_idx and pos
    cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, N_ELEMENTS))
    ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, 0))
    _cache = tl.load(CaChe + ori_seq_idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride)
    tl.store(output + idx * cache_stride + tl.arange(0, HIDDEN_DIM) * hidden_stride, _cache, mask=idx < total_length)


@triton.jit
def decoding_cache_kernel(
    CaChe,
    lengths,
    output,
    cache_stride,
    hidden_stride,
    HIDDEN_DIM: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ori_seq_idx = tl.load(lengths + idx, mask=(idx < NUM_SEQS), other=None)  # [BLOCK_SIZE,]
    _cache = tl.load(CaChe + ori_seq_idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride)
    tl.store(
        output + (idx[:, None] * cache_stride + tl.arange(0, HIDDEN_DIM)[None, :] * hidden_stride),
        _cache,
        mask=idx[:, None] < NUM_SEQS,
    )


@torch.no_grad()
def get_xine_cache(lengths: torch.Tensor, cache: torch.Tensor, is_prompts: bool = False):
    """
    Transform cos/sin cache into no pad sequence, with two different modes.
        Args:
            lengths: shape(num_seqs,), stores lenghth of each sequence.
            cache: shape(max_rotary_position(e.g.2048), head_dim), cos/sin cache constrcuted in model.
            is_prompts: bool, mark if in prefill mode.
        For prefill mode:
            cos/sin cache for each sequence is equal to its length.
        For decoding mode:
            cos/sin cache is only needed for the last token.
    """

    _, hidden_dim = cache.shape
    num_seqs = lengths.numel()

    BLOCK_SIZE = 16
    if hidden_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4

    cache_stride = cache.stride(0)
    hidden_stride = cache.stride(1)

    if is_prompts:
        total_length = lengths.sum().item()
        cumsum_lens = torch.cumsum(lengths, dim=0)
        output = torch.empty((total_length, hidden_dim), dtype=cache.dtype, device=cache.device)
        grid = (triton.cdiv(total_length, BLOCK_SIZE), BLOCK_SIZE)
        prefill_cache_kernel[grid](
            cache,
            cumsum_lens,
            output,
            cache_stride,
            hidden_stride,
            total_length,
            HIDDEN_DIM=hidden_dim,
            N_ELEMENTS=triton.next_power_of_2(num_seqs),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # BUG: get memory access error whe using a deepcopy lengths to replace lengths
        nlengths = torch.as_tensor(lengths) - 1
        output = torch.empty((num_seqs, hidden_dim), dtype=cache.dtype, device=cache.device)
        grid = (triton.cdiv(num_seqs, BLOCK_SIZE),)
        decoding_cache_kernel[grid](
            cache,
            nlengths,
            output,
            cache_stride,
            hidden_stride,
            HIDDEN_DIM=hidden_dim,
            NUM_SEQS=num_seqs,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return output
