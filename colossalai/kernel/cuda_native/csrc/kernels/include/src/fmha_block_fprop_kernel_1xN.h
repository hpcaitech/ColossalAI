/***************************************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "fmha_fprop_kernel_1xN.h"
#include "fmha_kernel.h"
#include "fmha_blockmask.h"
#include <fmha/kernel_traits.h>
#include <fmha/gemm.h>

namespace fmha {

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, bool Is_first, bool Is_last, typename Params, typename Prng>
inline __device__ void device_block_1xN_(const Params &params, const int bidb, const int bidh, int steps, Prng &ph0, Prng &ph1, const int loop_step_idx) {


    // The description of the CTA tile for the 1st batched GEMM.
    using Cta_tile_p = typename Kernel_traits::Cta_tile_p;
    // The description of the CTA tile for the 2nd batched GEMM.
    using Cta_tile_o = typename Kernel_traits::Cta_tile_o;

    // The MMA tile for the 1st GEMM.
    using Mma_tile_p = fmha::Hmma_tile<Cta_tile_p>;
    // The MMA tile for the 2nd GEMM.
    using Mma_tile_o = fmha::Hmma_tile<Cta_tile_o>;

    // The global memory tile to load Q.
    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;

    // The global memory tile to load K.
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;

    // The global memory tile to load V.
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;
    // The shared memory tile to swizzle V.
    using Smem_tile_v = typename Kernel_traits::Smem_tile_v;

    // The global memory tile to store O.
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_o_tmp = fmha::Gmem_tile_o<Cta_tile_o, 4>;
    // The shared memory tile to swizzle O.
    using Smem_tile_o = typename Kernel_traits::Smem_tile_o;

    using Gmem_tile_s = typename Kernel_traits::Gmem_tile_s;

    using Gmem_softmax_sum = typename Kernel_traits::Gmem_softmax_sum;

    using Smem_softmax_sum = typename Kernel_traits::Smem_dp_sum;

    using Gemm1 = Gemm_Q_K<Kernel_traits, Kernel_traits::K_IN_REGS>;

    using Softmax = fmha::Softmax<Cta_tile_p, Kernel_traits>;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    const BlockInfoPadded<Kernel_traits::THREADS> binfo(params, bidb, bidh, tidx);
    // if( binfo.stop_early() ) return;
    if( binfo.stop_early(loop_step_idx * Cta_tile_p::N) ) return;

    Blockmask blockmask(params, loop_step_idx);
    int block_row_idx = 0;
    int mask_val = blockmask.mask_val(0);
    if (mask_val == -1) return;
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("mask_val = %d.\n", mask_val);
    // }

    Gemm1 gemm_q_k(smem_, tidx);
    // Allocate the global memory tile loader for Q.
    Gmem_tile_q gmem_q(params.q_ptr, params.q_row_stride_in_elts, params.q_head_stride_in_elts,
                       params.d, binfo, tidx, true);
    // Allocate the global memory tile loader for O.
    Gmem_tile_o gmem_o(params.o_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                       params.d, binfo, tidx);
    Gmem_tile_o_tmp gmem_o_tmp(params.o_tmp_ptr, params.o_row_stride_in_elts, params.o_head_stride_in_elts,
                               params.d, binfo, tidx);
    // Allocate the global memory tile loader for S.
    Gmem_tile_s gmem_s(params, binfo, tidx);
    Gmem_softmax_sum gmem_softmax_lse(params.softmax_lse_ptr, params, tidx);

    // Wind gmem tiles to the correct position.
    static_assert(Cta_tile_p::N % Cta_tile_p::M == 0);
    int block_row_idx_next = mask_val / 4;
    int block_row_idx_to_move = block_row_idx_next - block_row_idx;
    gmem_q.move(block_row_idx_to_move);
    gmem_o.move(block_row_idx_to_move);
    gmem_o_tmp.move(block_row_idx_to_move);
    if (Return_softmax) { gmem_s.move(block_row_idx_to_move); }
    gmem_softmax_lse.move(block_row_idx_to_move);
    block_row_idx = block_row_idx_next;
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("begin = %d, steps = %d\n", begin, steps);
    // }

    fmha::Mask<Cta_tile_p, Is_causal> mask(binfo, tidx, loop_step_idx);

    // Allocate the global memory tile loader for K.
    Gmem_tile_k gmem_k(params.k_ptr, params.k_row_stride_in_elts, params.k_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // Allocate the global memory tile loader for V.
    Gmem_tile_v gmem_v(params.v_ptr, params.v_row_stride_in_elts, params.v_head_stride_in_elts,
                       params.d, binfo, tidx, false);
    // The base pointer of smem_v;
    char *smem_v_ = &smem_[Gemm1::SMEM_OFFSET_V];

    // Allocate the shared memory tile loader for V. We use the same as K so be careful!!!
    Smem_tile_v smem_v(smem_v_, tidx);

    // Allocate the shared memory tile loader for O. We use the same as K so be careful!!!
    Smem_tile_o smem_o(&smem_[Gemm1::SMEM_OFFSET_O], tidx);

    if (!Is_first) {
        gmem_k.move(loop_step_idx);
        gmem_v.move(loop_step_idx);
        if (Return_softmax) { gmem_s.move(loop_step_idx * steps); }
    }

    // Trigger the loads for K.
    gmem_k.load();
    // Trigger the loads for Q.
    gmem_q.load();
    // Trigger the loads for V.
    gmem_v.load();

    if (!Is_first) { __syncthreads(); }

    float p_prev_lse[Mma_tile_p::MMAS_M * 2];
    if (!(Is_first || mask_val % 2 == 1)) {
        gmem_softmax_lse.load(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse));
    }

    // Commit the data for Q and V to shared memory.
    gmem_q.commit(gemm_q_k.smem_q);
    gmem_v.commit(smem_v);

    // const uint32_t scale_bmm1 = reinterpret_cast<const uint32_t&>(params.scale_bmm1);
    // #pragma unroll
    // for(int it=0;it < Gmem_tile_k::LDGS;it++){
    //     gmem_k.fetch_[it] = fmha::hmul8(scale_bmm1, gmem_k.fetch_[it]);
    // }

    // Commit the data for K to shared memory.
    if( !Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        gmem_k.commit(gemm_q_k.smem_k);
    }

    __syncthreads();

    // Load the fragments for Q.
    gemm_q_k.load_q();

    // Load the fragments for V. We keep the data in registers during the entire kernel.
    typename Smem_tile_v::Fragment frag_v[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_N];
    #pragma unroll
    for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
        smem_v.load(frag_v[ki], ki);
    }

    // Commit the data for V to shared memory if it has not been done already.
    if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V ) {
        // Make sure we are done loading the fragments for K.
        __syncthreads();

        // Commit the data to shared memory for V.
        gmem_k.commit(gemm_q_k.smem_k);

        // Make sure the data is in shared memory.
        __syncthreads();
    }

    // Load the fragments for K.
    gemm_q_k.load_k();

    // Create the object to do the softmax.
    Softmax softmax(params, &smem_[Gemm1::SMEM_OFFSET_SOFTMAX], tidx);

    Smem_softmax_sum smem_softmax_lse(reinterpret_cast<float *>(&smem_[Gemm1::SMEM_BYTES]), tidx);

    // Load over the entire sequence length.
    for( int l = 0; l < steps; l++ ) {
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("block_row_idx = %d\n", block_row_idx);
        // }
        if (block_row_idx * Cta_tile_p::M >= binfo.actual_seqlen_q) break;

        int mask_val_next = l < steps - 1 ? blockmask.mask_val(l + 1) : -1;
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("mask_val = %d, mask_val_next = %d\n", mask_val, mask_val_next);
        // }

        // Declare the accumulators for the 1st gemm.
        fmha::Fragment_accumulator acc_p[Mma_tile_p::MMAS_M][Mma_tile_p::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_p::WARPS_K>::apply(acc_p);

        // Do this part of P = Q * K^T.
        gemm_q_k(acc_p);

        uint4 out[Gmem_tile_o::STGS_PER_LOOP];
        bool is_first_read = Is_first || mask_val % 2 == 1;
        // if (!Is_first) { gmem_o_tmp.load(out, 0); }
        if (!is_first_read) { gmem_o_tmp.load(out, 0); }

        // Trigger the load for the next Q values.
        bool not_last_iter = (l < steps - 1) && (mask_val_next != -1);
        block_row_idx_next = mask_val_next / 4;
        int block_row_idx_to_move = block_row_idx_next - block_row_idx;
        if (not_last_iter) {
            gemm_q_k.smem_q.move_to_next_write_buffer();
            gmem_q.move(block_row_idx_to_move);
            gmem_q.load();
        }

        // Load the mask for that iteration.
        mask.load(block_row_idx);

        // Convert from the accumulator type to FP32 for Softmax.
        softmax.unpack_noscale(acc_p);

        // Apply the mask.
        softmax.apply_mask(mask);

        // softmax.unpack_noscale_half_and_apply_mask(acc_p, mask);

        if( Kernel_traits::SHARE_SMEM_FOR_K_AND_V && l == 0 ) {
            // if we share K and V, it could be that V was not fully read yet but we write into smem for reduction
            __syncthreads();
        }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("p_prev_lse=%.6f, %.6f\n", p_prev_lse[0], p_prev_lse[1]);
        //     }
        // }
        // Compute the max.
        float p_max[Mma_tile_p::MMAS_M * 2];
        // if (!Is_first) {
        if (!is_first_read) {
            smem_softmax_lse.store_pair(p_prev_lse, l % 2);
            // for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi]; }
            for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) { p_max[mi] = p_prev_lse[mi] / params.scale_bmm1f; }
        }

        // Trigger the load for the next LSE values.
        if (not_last_iter) {
            // if (!Is_first) {
            if (!(Is_first || mask_val_next % 2 == 1)) {
                gmem_softmax_lse.load_next(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_prev_lse),
                                           block_row_idx_to_move);
            }
        }

        // __half2 p_max[Mma_tile_p::MMAS_M];
        // softmax.template reduce_max</*zero_init=*/Is_first>(p_max);
        is_first_read ? softmax.template reduce_max</*zero_init=*/true>(p_max) : softmax.template reduce_max</*zero_init=*/false>(p_max);

        // if ((threadIdx.x == 0) && (l == 38)) {
        //     printf("loop_step_idx %d, p_max = %.6f, %.6f., p_prev_lse = %.6f, %.6f\n", loop_step_idx, p_max[0], p_max[1], Is_first ? -10000.f : p_prev_lse[0], Is_first ? -10000.f : p_prev_lse[1]);
        // }

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after reduce_max=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the exponential value.
        // softmax.apply_exp(p_max);
        softmax.scale_apply_exp(p_max, params.scale_bmm1f);

        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("after apply_exp=%.6f, %.6f\n", softmax.elt_[0][0], softmax.elt_[0][1]);
        //     }
        // }

        // Compute the sum.
        float p_sum[Mma_tile_p::MMAS_M * 2];
        // if (!Is_first) {
        //     int warp = tidx / Cta_tile_p::THREADS_PER_WARP;
        //     int lane = tidx % Cta_tile_p::THREADS_PER_WARP;
        //     for (int mi = 0; mi < Mma_tile_p::MMAS_M * 2; mi++) {
        //         p_sum[mi] = ((warp == 0) && (lane % 4 == 0)) ? expf(p_prev_lse[mi] - p_max[mi]) : 0;
        //     }
        // }
        // softmax.reduce_sum(p_sum);
        softmax.reduce_sum_before_sync_(p_sum);
        // softmax.template reduce_sum_before_sync_</*zero_init=*/Is_first>(p_sum);

        // float p_sum_log[Mma_tile_p::MMAS_M * 2];
        // for (int mi = 0; mi  < Mma_tile_p::MMAS_M * 2; ++mi) {
        //     float sum = p_sum[mi];
        //     // p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] + __logf(sum);
        //     constexpr float kLog2e = M_LOG2E;
        //     p_sum_log[mi] = (sum == 0.f || sum != sum) ? INFINITY : p_max[mi] * kLog2e + __log2f(sum);
        // }
        // // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum));
        // gmem_softmax_lse.store(reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M * 2]>(p_sum_log));
        // gmem_softmax_lse.move();

        // // Finalize softmax on the accumulators of P^T.
        // softmax.scale(p_sum);

        constexpr bool encode_dropout_in_sign_bit = Return_softmax;
        if (Is_dropout) {
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph0, params.p_dropout_in_uint);
            // softmax.template apply_dropout<encode_dropout_in_sign_bit>(ph0, ph1, params.p_dropout_in_uint);
            softmax.template apply_dropout_16bits<encode_dropout_in_sign_bit>(ph0, ph1, params.p_dropout_in_uint16_t);
        }

        using Frag_p = fmha::Fragment_a<fmha::Row>;
        Frag_p frag_p[Mma_tile_o::MMAS_K][Mma_tile_o::MMAS_M];
        static_assert(Mma_tile_o::MMAS_M == Mma_tile_p::MMAS_M);
        static_assert(Mma_tile_o::MMAS_K == Mma_tile_p::MMAS_N);
        softmax.template pack<__half>(frag_p);
        if (Return_softmax) {
            gmem_s.store(frag_p, mask);
            if (not_last_iter) {
                gmem_s.move(block_row_idx_to_move);
            }
        }

        // Commit the values for Q into shared memory.
        if (not_last_iter) {
            gmem_q.commit(gemm_q_k.smem_q);
        }

        if (Is_dropout && encode_dropout_in_sign_bit) {
            #pragma unroll
            for( int ki = 0; ki < Mma_tile_o::MMAS_K; ki++ ) {
                #pragma unroll
                for( int mi = 0; mi < Mma_tile_o::MMAS_M; mi++ ) {
                    frag_p[ki][mi].template hrelu_<__half>();
                }
            }
        }

        // Declare the accumulators for the 2nd gemm.
        fmha::Fragment_accumulator acc_o[Mma_tile_o::MMAS_M][Mma_tile_o::MMAS_N];
        fmha::Clear_accumulator<typename fmha::Accumulator_type, Cta_tile_o::WARPS_K>::apply(acc_o);

        // Do this part of O = P^T * V^T.
        #pragma unroll
        for( int ki = 0; ki < Mma_tile_o::MMAS_K; ++ki ) {
            fmha::gemm_cl<__half>(acc_o, frag_p[ki], frag_v[ki]);
        }

        // The mapping from tidx to rows changes between the softmax and the O-reduction.
        // So we recalculate the max.
        float p_max_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        int rows[Gmem_tile_o::STGS_PER_LOOP];
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            rows[jj] = tidx / Gmem_tile_o::THREADS_PER_ROW + jj * Gmem_tile_o::ROWS_PER_STG;
        }
        softmax.reduce_max_after_sync_(p_max_o, rows);
        static_assert(Mma_tile_o::MMAS_M == 1);
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            p_max_o[jj][0] *= params.scale_bmm1f;
        }
        float p_prev_scale_o[Gmem_tile_o::STGS_PER_LOOP];
        // if (!Is_first) { smem_softmax_lse.load(p_prev_scale_o, rows, l % 2); }
        if (!is_first_read) { smem_softmax_lse.load(p_prev_scale_o, rows, l % 2); }
        // if (!Is_first) {
        //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
        //         printf("p_prev_scale_o=%.6f\n", p_prev_scale_o[0]);
        //     }
        // }

        static_assert(Gmem_tile_o::LOOPS == 1);

        // Swizzle the elements and do the final reduction.
        smem_o.store(acc_o, 0);

        // Make sure the data is in shared memory.
        __syncthreads();

        static_assert(Mma_tile_o::MMAS_M == 1);
        float p_sum_o[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        softmax.reduce_sum_after_sync_(p_sum_o, rows);
        // if (!Is_first) {
        if (!is_first_read) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                p_prev_scale_o[jj] = expf(p_prev_scale_o[jj] - p_max_o[jj][0]);
                p_sum_o[jj][0] += p_prev_scale_o[jj];
            }
        }

        float p_sum_log[Gmem_tile_o::STGS_PER_LOOP][Mma_tile_o::MMAS_M];
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            p_sum_log[jj][0] = (sum == 0.f || sum != sum) ? -INFINITY : p_max_o[jj][0] + __logf(sum);
            // if (sum == 0.f || sum != sum) {
            //     printf("loop_step_idx = %d, l = %d, tidx = %d, sum = %.6f, p_max_o = %.6f\n", loop_step_idx, l, tidx, sum, p_max_o[jj][0]);
            // }
            // if (Is_first) {
            //     if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0) && (l == 0))  {
            //         printf("p_sum_log=%.6f\n", p_sum_log[jj][0]);
            //     }
            // }
            if ((tidx % Gmem_tile_o::THREADS_PER_ROW == 0) && (tidx / Gmem_tile_o::THREADS_PER_ROW < Gmem_tile_o::ROWS)) {
                gmem_softmax_lse.store_row(
                    reinterpret_cast<uint32_t(&)[Mma_tile_p::MMAS_M]>(p_sum_log[jj]), rows[jj]);
            }
        }
        if (not_last_iter) {
            gmem_softmax_lse.move(block_row_idx_to_move);
        }

        // Load from shared memory.
        // if (!Is_first) {
        if (!is_first_read) {
            for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
                out[jj] = fmha::fmul4(out[jj], p_prev_scale_o[jj]);
            }
        }
        // smem_o.template load</*zero_init=*/Is_first>(out);
        is_first_read ? smem_o.template load</*zero_init=*/true>(out) : smem_o.template load</*zero_init=*/false>(out);

        const bool is_final_write =
            Is_last
            || ((loop_step_idx + 1) * Cta_tile_p::N >= binfo.actual_seqlen_k)
            || ((mask_val & 0x2) != 0)
            || ((Is_causal) && (block_row_idx * Cta_tile_p::M < (loop_step_idx + 1) * Cta_tile_p::N));
        // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        //     printf("is_final_write = %d\n", is_final_write);
        // }
        #pragma unroll
        for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
            float sum = p_sum_o[jj][0];
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if (Is_dropout && is_final_write) {
                inv_sum *= params.rp_dropout;
            }
            out[jj] = fmha::fmul4(out[jj], inv_sum);
        }

        // if (Is_dropout && Is_last) {
        //     for (int jj = 0; jj < Gmem_tile_o::STGS_PER_LOOP; jj++) {
        //         out[jj] = fmha::fmul4(out[jj], params.rp_dropout);
        //     }
        // }

        // Output the values.
        if (is_final_write) {
            gmem_o.template store<__half>(out, 0);
        } else {
            gmem_o_tmp.store(out, 0);
        }

        // Move to the next part of the output.
        gmem_o.move(block_row_idx_to_move);
        if (!(Is_first && Is_last)) { gmem_o_tmp.move(block_row_idx_to_move); }
        gemm_q_k.reload_k();

        // Make sure we are reading from the correct buffer.
        gemm_q_k.smem_q.move_to_next_read_buffer();
        // Trigger the load from shared memory for the next series of Q values.
        if (not_last_iter) {
            gemm_q_k.reload_q();
        }

        if (mask_val_next == -1) break;
        mask_val = mask_val_next;
        block_row_idx += block_row_idx_to_move;

    }  // Outer loop over the sequence length.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax, typename Params>
inline __device__ void device_block_1xN_loop(const Params &params) {

    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;
    // The thread index.
    const int tidx = threadIdx.x;

    const int tidx_global = (bidb * params.h + bidh) * blockDim.x * 2 + tidx;
    auto seeds = at::cuda::philox::unpack(params.philox_args);
    Philox ph0(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    Philox ph1(std::get<0>(seeds), tidx_global + blockDim.x, std::get<1>(seeds));
    constexpr int M = Kernel_traits::Cta_tile_p::M;
    const int STEPS = (params.seqlen_q + M - 1) / M;

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    if (params.seqlen_k == blocksize_c) {
        fmha::device_block_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, true>(params, bidb, bidh, STEPS, ph0, ph1, 0);
    } else {
        const int max_loop_steps = (params.seqlen_k + blocksize_c - 1) / blocksize_c;
        fmha::device_block_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, true, false>(params, bidb, bidh, STEPS, ph0, ph1, 0);
        for (int loop_step_idx = 1; loop_step_idx < max_loop_steps - 1; loop_step_idx++) {
            fmha::device_block_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, false>(params, bidb, bidh, STEPS, ph0, ph1, loop_step_idx);
        }
        fmha::device_block_1xN_<Kernel_traits, Is_dropout, Is_causal, Return_softmax, false, true>(params, bidb, bidh, STEPS, ph0, ph1, max_loop_steps - 1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

