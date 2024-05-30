/*This code adapted from vllm:
 *     https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "common/micros.h"
#include "funcs/cast_functor.h"
#include "funcs/ternary_functor.h"
#include "funcs/binary_functor.h"
#include "common/vec_type_traits.h"
#include "attention/attention_utils.h"

#define WARP_SIZE 32
#define PARTITION_SIZE 512
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
// 2^n => 2^n, 2^n-d => 2^(n-1)
#define ROUND_DOWN_HIGHEST_POWER_OF_TWO(x) (nextHighestPowerOf2((x - (x + 1) / 2 + 1)))

// a bit magic, you can ask chatgpt for help
// 2^n => 2^n, 2^n-d => 2^n
constexpr unsigned int nextHighestPowerOf2(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template <typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ii++) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

using colossalAI::funcs::BinaryOpType;
using colossalAI::funcs::CastFunctor;
using colossalAI::funcs::TernaryOpFunctor;
using colossalAI::funcs::TernaryOpType;
using colossalAI::common::VecTypeTrait;
using colossalAI::common::FloatVecTypeTrait;
using namespace colossalAI::cuda::attention;

template<typename scalar_t, typename KVecT, int VEC_SIZE, int Q_SHARED_SIZE, int NUM_VECS_PER_THREAD, int NUM_THREADS_PER_X, int NUM_ROWS_PER_ROUNDS, int NUM_VECS_PER_TOKEN, int x>
__device__ void data_load(
  const float4* q_ptr,
  float4* q_shared,
  scalar_t* q_shared_ptr,
  KVecT* q_vecs,            // query cached at register for qk_dot, should be constructed with reference to key cache's layout
  const int* block_table,
  int* block_table_shared,
  const int lane,
  const int max_num_blocks_per_seq
) {

  #pragma unroll
  for (int idx = threadIdx.x; idx < Q_SHARED_SIZE; idx += blockDim.x) {
    q_shared[idx] = q_ptr[idx];
  }

  #pragma unroll
  for (int idx = threadIdx.x; idx < max_num_blocks_per_seq; idx += blockDim.x) {
    block_table_shared[idx] = block_table[idx];
  }

  __syncthreads();

  // each warp access a whole block

  #pragma unroll
  for (int idx = lane, i = 0; idx < NUM_ROWS_PER_ROUNDS * NUM_VECS_PER_TOKEN; idx += WARP_SIZE, i += 1) {
    const int offset0 = idx / NUM_THREADS_PER_X / NUM_ROWS_PER_ROUNDS;
    const int offset1 = idx % NUM_THREADS_PER_X;
    q_vecs[i] = *reinterpret_cast<KVecT*>(q_shared_ptr + offset0 * x + offset1 * VEC_SIZE);
  }
}

template<typename scalar_t, typename cache_t, typename KVecT, typename KQuantVecT, int NUM_WARPS, int NUM_VECS_PER_THREAD, int BLOCK_SIZE, int NUM_ROWS_PER_ROUNDS, int NUM_VECS_PER_TOKEN, int NUM_THREADS_PER_X, int x, int VEC_SIZE>
__device__ void qk_gemv(
  const cache_t* __restrict__ k_cache,
  const KVecT (&q_vecs)[NUM_VECS_PER_THREAD], // Qk_dot needs NUM_VECS_PER_THREAD to do loop unrolling
  float* logits,                              // shared memory to cache Qk_dot results
  int* block_table_shared,
  const float alibi_slope,
  const int context_len,
  float &qk_max,
  const float scale,
  const int kv_head_idx,
  const int warp_idx,
  const int lane,
  const int thread_group_offset,
  const int start_block_idx,
  const int end_block_idx,
  const int start_token_idx,
  const int kv_block_stride,
  const int kv_head_stride) {

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table_shared[block_idx]);

    KVecT k_vecs[NUM_VECS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; i += NUM_ROWS_PER_ROUNDS) {
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                     + kv_head_idx * kv_head_stride
                                     + i * x;
      #pragma unroll
      for (int idx = lane, j = 0; idx < NUM_ROWS_PER_ROUNDS * NUM_VECS_PER_TOKEN; idx += WARP_SIZE, j += 1) {
        const int offset0 = idx / NUM_THREADS_PER_X / NUM_ROWS_PER_ROUNDS;
        const int offset1 = (idx / NUM_THREADS_PER_X) % NUM_ROWS_PER_ROUNDS;
        const int offset2 = idx % NUM_THREADS_PER_X;
        k_vecs[j] = CastFunctor<KQuantVecT, KVecT>()(*reinterpret_cast<const KQuantVecT*>(k_ptr + offset0 * BLOCK_SIZE * x + offset1 * x + offset2 * VEC_SIZE));
      }

      float qk = scale * Qk_dot<scalar_t, NUM_ROWS_PER_ROUNDS * NUM_THREADS_PER_X, NUM_THREADS_PER_X>::dot(q_vecs, k_vecs);

      if (thread_group_offset == 0 && lane < NUM_ROWS_PER_ROUNDS * NUM_THREADS_PER_X) {
        const int token_idx = block_idx * BLOCK_SIZE + i * NUM_ROWS_PER_ROUNDS + lane / NUM_THREADS_PER_X;
        qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
}

template<int NUM_THREADS, int NUM_WARPS, int NUM_ROWS_PER_ROUNDS, int NUM_THREADS_PER_X>
__device__ void softmax(
  float* red_shared_mem,
  float* logits,
  float &qk_max,
  float &exp_sum,
  int num_tokens) {
  // there exists a __syncthreads within this function
  qk_max = block_max<NUM_WARPS, NUM_ROWS_PER_ROUNDS * NUM_THREADS_PER_X, NUM_THREADS_PER_X>(red_shared_mem, qk_max);

  // Get the sum of the exp values.
  for (int i = threadIdx.x; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }

  exp_sum = block_sum<NUM_WARPS>(&red_shared_mem[NUM_WARPS], exp_sum);
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = threadIdx.x; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();
}

template<typename scalar_t, typename cache_t, typename FloatVecT, typename VVecT, typename VQuantVecT, int NUM_WARPS, int NUM_ROUNDS_PER_TOKEN, int NUM_THREADS_PER_TOKEN, int BLOCK_SIZE, int VEC_SIZE, int NUM_VECS_PER_TOKEN, int WARP_STRIDE>
__device__ void sv_gemv(
  const cache_t* __restrict__ v_cache,
  int* block_table_shared,
  float* out_shared_mem,      // shared memory to cache sv_gemv results
  float* logits,
  FloatVecT* accs,            // registers for accumulation
  const int lane,
  const int warp_idx,
  const int kv_head_idx,
  const int start_block_idx,
  const int end_block_idx,
  const int context_len,
  const int start_token_idx,
  const int kv_block_stride,
  const int kv_head_stride) {

  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    zero(accs[i]);
  }

  VVecT zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table_shared[block_idx]);
    scalar_t logit;

    #pragma unroll
    for (int idx = lane; idx < BLOCK_SIZE * NUM_VECS_PER_TOKEN; idx += WARP_STRIDE) {
      const int token_idx = block_idx * BLOCK_SIZE + idx / NUM_VECS_PER_TOKEN;
      const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                     + kv_head_idx * kv_head_stride
                                     + idx * VEC_SIZE;

      VVecT v_vecs[NUM_ROUNDS_PER_TOKEN];

      #pragma unroll
      for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
        v_vecs[i] = CastFunctor<VQuantVecT, VVecT>()(*((reinterpret_cast<const VQuantVecT*>(v_ptr) + i * WARP_SIZE)));
      }

      if (token_idx >= context_len) {
        #pragma unroll
        for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
          v_vecs[i] = zero_value;
        }
      }

      logit = CastFunctor<float, scalar_t>()(logits[token_idx - start_token_idx]);
      #pragma unroll
      for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
        accs[i] = TernaryOpFunctor<scalar_t, VVecT, FloatVecT, TernaryOpType::kFma>()(logit, v_vecs[i], accs[i]);
      }
    }
  }

  // must insert a sync since both logits and out_shared_mem occupy the same buffer space
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    block_sum<FloatVecT, NUM_WARPS, NUM_THREADS_PER_TOKEN, VEC_SIZE>(out_shared_mem, accs[i]);
  }
}

// We only support head size of { 64, 128, 256 }
// models like Phi-2, whose head size is 80, is not supported right now
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void flash_decoding_attention_kernel_v1(
  scalar_t* __restrict__ out,                 // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ q,             // [num_tokens, num_heads, head_size]
  const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
  const int* __restrict__ context_lens,       // [num_tokens]
  const int* __restrict__ block_tables,       // [num_tokens, max_num_blocks_per_seq]
  const float* __restrict__ alibi_slopes,     // [num_heads]
  const int max_seq_len,
  const int num_kv_heads,
  const float scale,
  const int max_num_blocks_per_seq,
  const int q_stride,                         // num_heads * head_size
  const int kv_block_stride,
  const int kv_head_stride) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int lane = thread_idx % WARP_SIZE;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int x = sizeof(float4) / sizeof(scalar_t);
  constexpr int Q_SHARED_SIZE = HEAD_SIZE / x;
  // here thread_group does not determine the number of threads responsible for a key
  // but only the VEC_SIZE of each thread
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int VEC_SIZE = MIN(ROUND_DOWN_HIGHEST_POWER_OF_TWO((HEAD_SIZE / THREAD_GROUP_SIZE)), x);
  constexpr int NUM_VECS_PER_TOKEN = HEAD_SIZE / VEC_SIZE;
  constexpr int NUM_THREADS_PER_TOKEN = MIN(NUM_VECS_PER_TOKEN, WARP_SIZE);
  constexpr int NUM_ROUNDS_PER_TOKEN = NUM_VECS_PER_TOKEN / NUM_THREADS_PER_TOKEN;
  constexpr int WARP_STRIDE = WARP_SIZE * NUM_ROUNDS_PER_TOKEN;
  constexpr int NUM_THREADS_PER_X = x / VEC_SIZE;
  constexpr int NUM_ROWS_PER_ROUNDS = MIN(WARP_SIZE / NUM_THREADS_PER_X, BLOCK_SIZE);
  constexpr int NUM_VECS_PER_THREAD = NUM_ROWS_PER_ROUNDS * NUM_VECS_PER_TOKEN / WARP_SIZE;

  using KVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using VVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using KQuantVecT = typename VecTypeTrait<cache_t, VEC_SIZE>::Type;
  using VQuantVecT = typename VecTypeTrait<cache_t, VEC_SIZE>::Type;
  using LVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using FloatVecT = typename FloatVecTypeTrait<LVecT>::Type;

  const int context_len = context_lens[seq_idx];
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  const int thread_group_offset = lane % NUM_THREADS_PER_X;
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int shared_memory_offset = DIVIDE_ROUND_UP(max_num_blocks_per_seq * sizeof(int), sizeof(float4)) * sizeof(float4);

  __shared__ float4 q_shared[Q_SHARED_SIZE];
  __shared__ float red_shared_mem[2 * NUM_WARPS];
  extern __shared__ char shared_mem[];
  int* block_table_shared = reinterpret_cast<int*>(shared_mem);
  float* logits = reinterpret_cast<float*>(shared_mem + shared_memory_offset);
  float* out_shared_mem = reinterpret_cast<float*>(shared_mem + shared_memory_offset);
  float qk_max = -FLT_MAX;
  float exp_sum = 0.f;

  const float4* q_ptr = reinterpret_cast<const float4*>(q + seq_idx * q_stride + head_idx * HEAD_SIZE);
  scalar_t* q_shared_ptr = reinterpret_cast<scalar_t*>(q_shared);
  KVecT q_vecs[NUM_VECS_PER_THREAD];

  // 1. load query and block_table from global memory to shared memory
  data_load<scalar_t, KVecT, VEC_SIZE, Q_SHARED_SIZE, NUM_VECS_PER_THREAD, NUM_THREADS_PER_X, NUM_ROWS_PER_ROUNDS, NUM_VECS_PER_TOKEN, x>(q_ptr, q_shared, q_shared_ptr, q_vecs, block_table, block_table_shared, lane, max_num_blocks_per_seq);

  // 2. compute the dot product of query and key cache
  qk_gemv<scalar_t, cache_t, KVecT, KQuantVecT, NUM_WARPS, NUM_VECS_PER_THREAD, BLOCK_SIZE, NUM_ROWS_PER_ROUNDS, NUM_VECS_PER_TOKEN, NUM_THREADS_PER_X, x, VEC_SIZE>(k_cache, q_vecs, logits, block_table_shared, alibi_slope, context_len, qk_max, scale, kv_head_idx, warp_idx, lane, thread_group_offset, 0, num_context_blocks, 0, kv_block_stride, kv_head_stride);

  // 3. compute the softmax
  softmax<NUM_THREADS, NUM_WARPS, NUM_ROWS_PER_ROUNDS, NUM_THREADS_PER_X>(red_shared_mem, logits, qk_max, exp_sum, context_len);

  FloatVecT accs[NUM_ROUNDS_PER_TOKEN];

  // 4. compute the dot product of softmax tensor and value cache
  sv_gemv<scalar_t, cache_t, FloatVecT, VVecT, VQuantVecT, NUM_WARPS, NUM_ROUNDS_PER_TOKEN, NUM_THREADS_PER_TOKEN, BLOCK_SIZE, VEC_SIZE, NUM_VECS_PER_TOKEN, WARP_STRIDE>(v_cache, block_table_shared, out_shared_mem, logits, accs, lane, warp_idx, kv_head_idx, 0, num_context_blocks, context_len, 0, kv_block_stride, kv_head_stride);

  // 5. write back to global memory
  scalar_t* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
  LVecT out_reg;
  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    if (thread_idx < NUM_THREADS_PER_TOKEN) {
      out_reg = CastFunctor<FloatVecT, LVecT>()(accs[i]);
      (reinterpret_cast<LVecT*>(out_ptr))[thread_idx + i * NUM_THREADS_PER_TOKEN] = out_reg;
    }
  }
}

#define LAUNCH_FLASH_DECODING_ATTENTION_V1(HEAD_SIZE)                                            \
  cudaFuncSetAttribute(                                                                          \
    ((void*)flash_decoding_attention_kernel_v1<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>), \
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);                               \
  flash_decoding_attention_kernel_v1<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>             \
                       <<<grid, block, shared_mem_size, stream>>>(                               \
    reinterpret_cast<T*>(out.data_ptr()),                                                        \
    reinterpret_cast<T*>(query.data_ptr()),                                                      \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                            \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                          \
    context_lens.data_ptr<int>(),                                                                \
    block_tables.data_ptr<int>(),                                                                \
    alibi_slopes_ptr,                                                                            \
    max_context_len,                                                                             \
    num_kv_heads,                                                                                \
    scale,                                                                                       \
    max_num_blocks_per_seq,                                                                      \
    q_stride,                                                                                    \
    kv_block_stride,                                                                             \
    kv_head_stride);

template<
  typename T,
  typename CACHE_T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void flash_decoding_attention_v1_launcher(
  torch::Tensor& out,              // [num_tokens, num_heads, head_size]
  torch::Tensor& query,            // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,      // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& context_lens,     // [num_tokens]
  torch::Tensor& block_tables,     // [num_tokens, max_num_blocks_per_seq]
  int max_context_len,
  float scale,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_tokens = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int q_stride = query.stride(0);

  int max_num_blocks_per_seq = block_tables.size(1);

  int num_kv_heads = key_cache.size(1);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  const int VEC_SIZE = MIN(ROUND_DOWN_HIGHEST_POWER_OF_TWO((head_size / THREAD_GROUP_SIZE)), sizeof(float4) / sizeof(T));
  const int NUM_VECS_PER_TOKEN = head_size / VEC_SIZE;
  const int NUM_THREADS_PER_TOKEN = MIN(NUM_VECS_PER_TOKEN, WARP_SIZE);

  int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * NUM_THREADS_PER_TOKEN * VEC_SIZE * sizeof(float);
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size) + DIVIDE_ROUND_UP(max_num_blocks_per_seq * sizeof(int), sizeof(float4)) * sizeof(float4);

  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  dim3 grid(num_heads, num_tokens, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model.
    case 64:
      LAUNCH_FLASH_DECODING_ATTENTION_V1(64);
      break;
    case 128:
      LAUNCH_FLASH_DECODING_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_FLASH_DECODING_ATTENTION_V1(256);
      break;
    default:
      AT_ERROR("head size must be 64, 128, 256");
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE)                             \
  flash_decoding_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE>(              \
    out,                                                                     \
    query,                                                                   \
    key_cache,                                                               \
    value_cache,                                                             \
    context_lens,                                                            \
    block_tables,                                                            \
    max_context_len,                                                         \
    scale,                                                                   \
    alibi_slopes);


template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void flash_decoding_attention_kernel_v2(
  scalar_t* __restrict__ out,                 // [num_tokens, num_heads, max_num_partitions, head_size]
  float* __restrict__ exp_sums,               // [num_tokens, num_heads, max_num_partitions]
  float* __restrict__ max_logits,             // [num_tokens, num_heads, max_num_partitions]
  const scalar_t* __restrict__ q,             // [num_tokens, num_heads, head_size]
  const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
  const int* __restrict__ context_lens,       // [num_tokens]
  const int* __restrict__ block_tables,       // [num_tokens, max_num_blocks_per_seq]
  const float* __restrict__ alibi_slopes,     // [num_heads]
  const int max_seq_len,
  const int num_kv_heads,
  const float scale,
  const int max_num_blocks_per_seq,
  const int q_stride,                         // num_heads * head_size
  const int tmp_stride,                       // num_heads * max_num_partitions
  const int kv_block_stride,
  const int kv_head_stride) {
  const int partition_idx = blockIdx.z;
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int thread_idx = threadIdx.x;
  const int lane = thread_idx % WARP_SIZE;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int max_num_partitions = gridDim.z;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int x = sizeof(float4) / sizeof(scalar_t);
  constexpr int Q_SHARED_SIZE = HEAD_SIZE / x;
  // here thread_group does not determine the number of threads responsible for a key
  // but only the VEC_SIZE of each thread
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int VEC_SIZE = MIN(ROUND_DOWN_HIGHEST_POWER_OF_TWO((HEAD_SIZE / THREAD_GROUP_SIZE)), x);
  constexpr int NUM_VECS_PER_TOKEN = HEAD_SIZE / VEC_SIZE;
  constexpr int NUM_THREADS_PER_TOKEN = MIN(NUM_VECS_PER_TOKEN, WARP_SIZE);
  constexpr int NUM_ROUNDS_PER_TOKEN = NUM_VECS_PER_TOKEN / NUM_THREADS_PER_TOKEN;
  constexpr int WARP_STRIDE = WARP_SIZE * NUM_ROUNDS_PER_TOKEN;
  constexpr int NUM_THREADS_PER_X = x / VEC_SIZE;
  constexpr int NUM_ROWS_PER_ROUNDS = MIN(WARP_SIZE / NUM_THREADS_PER_X, BLOCK_SIZE);
  constexpr int NUM_VECS_PER_THREAD = NUM_ROWS_PER_ROUNDS * NUM_VECS_PER_TOKEN / WARP_SIZE;
  constexpr int NUM_BLOCKS_PER_PARTITION = PARTITION_SIZE / BLOCK_SIZE;

  using KVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using VVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using KQuantVecT = typename VecTypeTrait<cache_t, VEC_SIZE>::Type;
  using VQuantVecT = typename VecTypeTrait<cache_t, VEC_SIZE>::Type;
  using LVecT = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using FloatVecT = typename FloatVecTypeTrait<LVecT>::Type;

  const int context_len = context_lens[seq_idx];

  if (partition_idx * PARTITION_SIZE >= context_len) {
    return;
  }

  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  const int thread_group_offset = lane % NUM_THREADS_PER_X;
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = partition_idx * NUM_BLOCKS_PER_PARTITION;
  const int end_block_idx = MIN(start_block_idx + NUM_BLOCKS_PER_PARTITION, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int shared_memory_offset = DIVIDE_ROUND_UP(max_num_blocks_per_seq * sizeof(int), sizeof(float4)) * sizeof(float4);

  __shared__ float4 q_shared[Q_SHARED_SIZE];
  __shared__ float red_shared_mem[2 * NUM_WARPS];
  extern __shared__ char shared_mem[];
  int* block_table_shared = reinterpret_cast<int*>(shared_mem);
  float* logits = reinterpret_cast<float*>(shared_mem + shared_memory_offset);
  float* out_shared_mem = reinterpret_cast<float*>(shared_mem + shared_memory_offset);
  float qk_max = -FLT_MAX;
  float exp_sum = 0.f;

  const float4* q_ptr = reinterpret_cast<const float4*>(q + seq_idx * q_stride + head_idx * HEAD_SIZE);
  scalar_t* q_shared_ptr = reinterpret_cast<scalar_t*>(q_shared);
  KVecT q_vecs[NUM_VECS_PER_THREAD];

  // 1. load query and block_table from global memory to shared memory
  data_load<scalar_t, KVecT, VEC_SIZE, Q_SHARED_SIZE, NUM_VECS_PER_THREAD, NUM_THREADS_PER_X, NUM_ROWS_PER_ROUNDS, NUM_VECS_PER_TOKEN, x>(q_ptr, q_shared, q_shared_ptr, q_vecs, block_table, block_table_shared, lane, max_num_blocks_per_seq);

  // 2. compute the dot product of query and key cache
  qk_gemv<scalar_t, cache_t, KVecT, KQuantVecT, NUM_WARPS, NUM_VECS_PER_THREAD, BLOCK_SIZE, NUM_ROWS_PER_ROUNDS, NUM_VECS_PER_TOKEN, NUM_THREADS_PER_X, x, VEC_SIZE>(k_cache, q_vecs, logits, block_table_shared, alibi_slope, context_len, qk_max, scale, kv_head_idx, warp_idx, lane, thread_group_offset, start_block_idx, end_block_idx, start_token_idx, kv_block_stride, kv_head_stride);

  // 3. compute the softmax
  softmax<NUM_THREADS, NUM_WARPS, NUM_ROWS_PER_ROUNDS, NUM_THREADS_PER_X>(red_shared_mem, logits, qk_max, exp_sum, num_tokens);

  if (thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * tmp_stride
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    float* exp_sums_ptr = exp_sums + seq_idx * tmp_stride
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *max_logits_ptr = qk_max;
    *exp_sums_ptr = exp_sum;
  }

  FloatVecT accs[NUM_ROUNDS_PER_TOKEN];

  // 4. compute the dot product of softmax tensor and value cache
  sv_gemv<scalar_t, cache_t, FloatVecT, VVecT, VQuantVecT, NUM_WARPS, NUM_ROUNDS_PER_TOKEN, NUM_THREADS_PER_TOKEN, BLOCK_SIZE, VEC_SIZE, NUM_VECS_PER_TOKEN, WARP_STRIDE>(v_cache, block_table_shared, out_shared_mem, logits, accs, lane, warp_idx, kv_head_idx, start_block_idx, end_block_idx, context_len, start_token_idx, kv_block_stride, kv_head_stride);

  // 5. write back to global memory
  scalar_t* out_ptr = out + seq_idx * q_stride * max_num_partitions
                          + head_idx * HEAD_SIZE * max_num_partitions
                          + partition_idx * HEAD_SIZE;
  LVecT out_reg;
  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    if (thread_idx < NUM_THREADS_PER_TOKEN) {
      out_reg = CastFunctor<FloatVecT, LVecT>()(accs[i]);
      (reinterpret_cast<LVecT*>(out_ptr))[thread_idx + i * NUM_THREADS_PER_TOKEN] = out_reg;
    }
  }
}

template<typename scalar_t, int HEAD_SIZE, int NUM_THREADS>
__global__ void flash_decoding_reduce_kernel(
  scalar_t* __restrict__ out,                 // [num_tokens, num_heads, head_size]
  float* __restrict__ exp_sums,               // [num_tokens, num_heads, max_num_partitions]
  float* __restrict__ max_logits,             // [num_tokens, num_heads, max_num_partitions]
  scalar_t* __restrict__ tmp_out,             // [num_tokens, num_heads, max_num_partitions, head_size]
  const int* __restrict__ context_lens,       // [num_tokens]
  const int out_stride,
  const int tmp_stride,
  const int max_num_partitions) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;

  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  extern __shared__ char shared_mem[];
  __shared__ float red_smem[2 * NUM_WARPS];
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * tmp_stride
                                           + head_idx * max_num_partitions;

  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float tmp_max_logit = max_logits_ptr[i];
    shared_max_logits[i] = tmp_max_logit;
    max_logit = fmaxf(max_logit, tmp_max_logit);
  }

  __syncthreads();

  max_logit = block_max<NUM_WARPS, WARP_SIZE, 1>(red_smem, max_logit);

  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + num_partitions * sizeof(float));
  const float* exp_sums_ptr = exp_sums + seq_idx * tmp_stride
                                       + head_idx * max_num_partitions;

  float global_exp_sum = 0.f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float tmp_max_logit = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(tmp_max_logit - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }

  __syncthreads();

  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.f, global_exp_sum + 1e-6f);

  const scalar_t* tmp_out_ptr = tmp_out + seq_idx * out_stride * max_num_partitions
                                        + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * out_stride + head_idx * HEAD_SIZE;

  #pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.f;
    for (int j = 0; j < num_partitions; j++) {
      acc += CastFunctor<scalar_t, float>()(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    out_ptr[i] = CastFunctor<float, scalar_t>()(acc);
  }
}


#define LAUNCH_FLASH_DECODING_ATTENTION_V2(HEAD_SIZE)                                            \
  cudaFuncSetAttribute(                                                                          \
    ((void*)flash_decoding_attention_kernel_v2<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>), \
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);                               \
  flash_decoding_attention_kernel_v2<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>             \
                       <<<grid, block, shared_mem_size, stream>>>(                               \
    reinterpret_cast<T*>(tmp_out.data_ptr()),                                                    \
    reinterpret_cast<float*>(exp_sums.data_ptr()),                                               \
    reinterpret_cast<float*>(max_logits.data_ptr()),                                             \
    reinterpret_cast<T*>(query.data_ptr()),                                                      \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                            \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                          \
    reinterpret_cast<int*>(context_lens.data_ptr()),                                             \
    reinterpret_cast<int*>(block_tables.data_ptr()),                                             \
    alibi_slopes_ptr,                                                                            \
    max_context_len,                                                                             \
    num_kv_heads,                                                                                \
    scale,                                                                                       \
    max_num_blocks_per_seq,                                                                      \
    q_stride,                                                                                    \
    tmp_stride,                                                                                  \
    kv_block_stride,                                                                             \
    kv_head_stride);                                                                             \
  cudaFuncSetAttribute(                                                                          \
    ((void*)flash_decoding_reduce_kernel<T, HEAD_SIZE, NUM_THREADS>),                            \
    cudaFuncAttributeMaxDynamicSharedMemorySize, reduce_shared_mem_size);                        \
  flash_decoding_reduce_kernel<T, HEAD_SIZE, NUM_THREADS>                                        \
                       <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                 \
    reinterpret_cast<T*>(out.data_ptr()),                                                        \
    reinterpret_cast<float*>(exp_sums.data_ptr()),                                               \
    reinterpret_cast<float*>(max_logits.data_ptr()),                                             \
    reinterpret_cast<T*>(tmp_out.data_ptr()),                                                    \
    reinterpret_cast<int*>(context_lens.data_ptr()),                                             \
    q_stride,                                                                                    \
    tmp_stride,                                                                                  \
    max_num_partitions);


template<
  typename T,
  typename CACHE_T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void flash_decoding_attention_v2_launcher(
  torch::Tensor& out,              // [num_tokens, num_heads, head_size]
  torch::Tensor& exp_sums,         // [num_tokens, num_heads, max_num_partitions]
  torch::Tensor& max_logits,       // [num_tokens, num_heads, max_num_partitions]
  torch::Tensor& tmp_out,          // [num_tokens, num_heads, max_num_partitions, head_size]
  torch::Tensor& query,            // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,        // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,      // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& context_lens,     // [num_tokens]
  torch::Tensor& block_tables,     // [num_tokens, max_num_blocks_per_seq]
  int max_context_len,
  float scale,
  const c10::optional<torch::Tensor>& alibi_slopes) {
  int num_tokens = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int q_stride = query.stride(0);
  int tmp_stride = exp_sums.stride(0);

  int max_num_blocks_per_seq = block_tables.size(1);

  int num_kv_heads = key_cache.size(1);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  const int VEC_SIZE = MIN(ROUND_DOWN_HIGHEST_POWER_OF_TWO((head_size / THREAD_GROUP_SIZE)), sizeof(float4) / sizeof(T));
  const int NUM_VECS_PER_TOKEN = head_size / VEC_SIZE;
  const int NUM_THREADS_PER_TOKEN = MIN(NUM_VECS_PER_TOKEN, WARP_SIZE);

  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * NUM_THREADS_PER_TOKEN * VEC_SIZE * sizeof(float);
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size) + DIVIDE_ROUND_UP(max_num_blocks_per_seq * sizeof(int), sizeof(float4)) * sizeof(float4);

  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  dim3 grid(num_heads, num_tokens, max_num_partitions);
  dim3 block(NUM_THREADS);

  dim3 reduce_grid(num_heads, num_tokens);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model.
    case 64:
      LAUNCH_FLASH_DECODING_ATTENTION_V2(64);
      break;
    case 128:
      LAUNCH_FLASH_DECODING_ATTENTION_V2(128);
      break;
    case 256:
      LAUNCH_FLASH_DECODING_ATTENTION_V2(256);
      break;
    default:
      AT_ERROR("head size must be 64, 128, 256");
      break;
  }
}

#define CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE)                 \
  flash_decoding_attention_v2_launcher<T, CACHE_T, BLOCK_SIZE>(  \
    out,                                                         \
    exp_sums,                                                    \
    max_logits,                                                  \
    tmp_out,                                                     \
    query,                                                       \
    key_cache,                                                   \
    value_cache,                                                 \
    context_lens,                                                \
    block_tables,                                                \
    max_context_len,                                             \
    scale,                                                       \
    alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_LAUNCHER_BLOCK_SIZE(Version, T, CACHE_T)                 \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_##Version##_LAUNCHER(T, CACHE_T, 8);                       \
      break;                                                          \
    case 16:                                                          \
      CALL_##Version##_LAUNCHER(T, CACHE_T, 16);                      \
      break;                                                          \
    case 32:                                                          \
      CALL_##Version##_LAUNCHER(T, CACHE_T, 32);                      \
      break;                                                          \
    default:                                                          \
      AT_ERROR("block size must be 8, 16, 32");                       \
      break;                                                          \
  }

#define CALL_LAUNCHER_DTYPE(Version)                                            \
  if(key_cache.scalar_type() == at::ScalarType::Byte)                           \
  {                                                                             \
    switch (query.scalar_type()) {                                              \
      case at::ScalarType::Float:                                               \
        CALL_LAUNCHER_BLOCK_SIZE(Version, float, uint8_t);                      \
        break;                                                                  \
      case at::ScalarType::Half:                                                \
        CALL_LAUNCHER_BLOCK_SIZE(Version, half, uint8_t);                       \
        break;                                                                  \
      case at::ScalarType::BFloat16:                                            \
        CALL_LAUNCHER_BLOCK_SIZE(Version, __nv_bfloat16, uint8_t);              \
        break;                                                                  \
    }                                                                           \
  }                                                                             \
  else                                                                          \
  {                                                                             \
    switch (query.scalar_type()) {                                              \
      case at::ScalarType::Float:                                               \
        CALL_LAUNCHER_BLOCK_SIZE(Version, float, float);                        \
        break;                                                                  \
      case at::ScalarType::Half:                                                \
        CALL_LAUNCHER_BLOCK_SIZE(Version, half, half);                          \
        break;                                                                  \
      case at::ScalarType::BFloat16:                                            \
        CALL_LAUNCHER_BLOCK_SIZE(Version, __nv_bfloat16, __nv_bfloat16);        \
        break;                                                                  \
    }                                                                           \
  }

void flash_decoding_attention(
  torch::Tensor& out,             // [num_tokens, num_heads, head_size]
  torch::Tensor& query,           // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& context_lens,    // [num_tokens]
  torch::Tensor& block_tables,    // [num_tokens, max_num_blocks_per_seq]
  int block_size,
  int max_context_len,
  torch::Tensor& tmp_out,         // [num_tokens, num_heads, max_num_partitions, head_size]
  torch::Tensor& exp_sums,        // [num_tokens, num_heads, max_num_partitions]
  torch::Tensor& max_logits,      // [num_tokens, num_heads, max_num_partitions]
  const c10::optional<torch::Tensor>& alibi_slopes,
  float scale) {

  int num_tokens = query.size(0);
  int num_heads = query.size(1);

  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  // TODO(luoxiang): Need to be tuned
  bool use_v1 = max_context_len <= 8192 && (max_num_partitions == 1 || num_tokens * num_heads > 512);

  if (use_v1) {
    CALL_LAUNCHER_DTYPE(V1);
  } else {
    CALL_LAUNCHER_DTYPE(V2);
  }
}


#undef LAUNCH_FLASH_DECODING_ATTENTION_V1
#undef CALL_LAUNCHER
#undef CALL_LAUNCHER_BLOCK_SIZE
#undef CALL_LAUNCHER_DTYPE
