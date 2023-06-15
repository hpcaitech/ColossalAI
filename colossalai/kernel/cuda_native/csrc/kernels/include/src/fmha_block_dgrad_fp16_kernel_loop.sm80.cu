/* Copyright (c) 2022, Tri Dao.
 */

#include "fmha.h"
#include "fmha_block_dgrad_kernel_1xN_loop.h"

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, int loop_steps=-1>
__global__ void fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel(FMHA_dgrad_params params) {
    fmha::compute_block_dq_dk_dv_1xN<Kernel_traits, Is_dropout, Is_causal, loop_steps>(params);
}

template<typename Kernel_traits>
void run_fmha_block_dgrad_fp16_sm80_loop_(const FMHA_dgrad_params &params, cudaStream_t stream) {
    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_dq = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;
    constexpr int smem_size_dp_sum = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * Kernel_traits::Cta_tile_p::N * 2);
    static_assert(smem_size_dq == 16 * Kernel_traits::Cta_tile_p::K * 4 * Kernel_traits::Cta_tile_p::WARPS_N);
    static_assert(smem_size_dp_sum == 16 * 4 * 2);

    constexpr int smem_size_dq_dk_dv = smem_size_q * 2 + smem_size_v * (Kernel_traits::V_IN_REGS ? 1 : 2) + smem_size_dq + smem_size_s * 2 + smem_size_dp_sum;

    bool is_dropout = params.p_dropout < 1.f;  // params.p_dropout is the probability of "keeping"
    bool is_causal = params.is_causal;
    auto kernel = is_dropout
        ? (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false>)
        : (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false>);
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    if (params.seqlen_k == blocksize_c) {
        kernel = is_dropout
            ? (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true, /*loop_steps=*/1> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false, /*loop_steps=*/1>)
            : (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true, /*loop_steps=*/1> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false, /*loop_steps=*/1>);
    } else if (params.seqlen_k == blocksize_c * 2) {
        kernel = is_dropout
            ? (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true, /*loop_steps=*/2> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false, /*loop_steps=*/2>)
            : (is_causal ? &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true, /*loop_steps=*/2> : &fmha_block_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false, /*loop_steps=*/2>);
    }

    if( smem_size_dq_dk_dv >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
    }
    dim3 grid(params.b, params.h);
    kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_block_dgrad_fp16_sm80(const FMHA_dgrad_params &params, cudaStream_t stream) {
    if (params.d == 16) {
        using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u>;
        run_fmha_block_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
    } else if (params.d == 32) {
        using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u>;
        run_fmha_block_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
    } else if (params.d == 64) {
        using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u>;
        run_fmha_block_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
    }
}