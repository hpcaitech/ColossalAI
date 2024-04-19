/*This code adapted from vllm:
 *     https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cu
 *     with different kvcache layout. */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>

#include "common/micros.h"
#include "funcs/cast_functor.h"
#include "funcs/ternary_functor.h"
#include "funcs/binary_functor.h"
#include "common/vec_type_traits.h"
#include "attention/attention_utils.h"

#define WARP_SIZE 32
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

using colossalAI::funcs::BinaryOpType;
using colossalAI::funcs::CastFunctor;
using colossalAI::funcs::TernaryOpFunctor;
using colossalAI::funcs::TernaryOpType;
using colossalAI::funcs::zero;
using colossalAI::common::VecTypeTrait;
using colossalAI::common::FloatVecTypeTrait;
using namespace colossalAI::cuda::attention;


// We only support head size of { 64, 128, 256 }
// models like Phi-2, whose head size is 80, is not supported right now
template<typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void flash_decoding_attention_kernel(
  scalar_t* __restrict__ out,                 // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ q,             // [num_tokens, num_heads, head_size]
  const cache_t* __restrict__ k_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
  const cache_t* __restrict__ v_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
  const int* __restrict__ context_lens,       // [num_tokens]
  const int* __restrict__ block_tables,       // [num_tokens, max_num_blocks_per_seq]
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
  constexpr int Q_SHARED_SIZE = (HEAD_SIZE * sizeof(scalar_t)) / sizeof(float4);
  // here thread_group does not determine the number of threads responsible for a key
  // but only the VEC_SIZE of each thread
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int VEC_SIZE = MIN(ROUND_DOWN_HIGHEST_POWER_OF_TWO((HEAD_SIZE / THREAD_GROUP_SIZE)), sizeof(float4) / sizeof(scalar_t));
  constexpr int NUM_VECS_PER_TOKEN = HEAD_SIZE / VEC_SIZE;
  constexpr int NUM_THREADS_PER_TOKEN = MIN(NUM_VECS_PER_TOKEN, WARP_SIZE);
  constexpr int NUM_ROUNDS_PER_TOKEN = NUM_VECS_PER_TOKEN / NUM_THREADS_PER_TOKEN;
  constexpr int WARP_STRIDE = WARP_SIZE * NUM_ROUNDS_PER_TOKEN;

  using K_vec = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using V_vec = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using L_vec = typename VecTypeTrait<scalar_t, VEC_SIZE>::Type;
  using Float_vec = typename FloatVecTypeTrait<L_vec>::Type;

  const int context_len = context_lens[seq_idx];
  const int thread_group_offset = thread_idx % NUM_THREADS_PER_TOKEN;
  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  __shared__ float4 q_shared[Q_SHARED_SIZE];
  __shared__ float red_shared_mem[2 * NUM_WARPS];
  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  float* out_shared_mem = reinterpret_cast<float*>(shared_mem);
  float qk_max = -FLT_MAX;

  const float4* q_ptr = reinterpret_cast<const float4*>(q + seq_idx * q_stride + head_idx * HEAD_SIZE);
  #pragma unroll
  for (int idx = thread_idx; idx < Q_SHARED_SIZE; idx += blockDim.x) {
    q_shared[idx] = q_ptr[idx];
  }
  __syncthreads();

  scalar_t* q_shared_ptr = reinterpret_cast<scalar_t*>(q_shared);
  // each warp access a whole block
  for (int block_idx = warp_idx; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    #pragma unroll
    for (int idx = lane; idx < BLOCK_SIZE * NUM_VECS_PER_TOKEN; idx += WARP_STRIDE) {
      const int token_idx = block_idx * BLOCK_SIZE + idx / NUM_VECS_PER_TOKEN;
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                                     + kv_head_idx * kv_head_stride
                                     + idx * VEC_SIZE;

      K_vec k_vecs[NUM_ROUNDS_PER_TOKEN];
      K_vec q_vecs[NUM_ROUNDS_PER_TOKEN];

      // we must calculate at least one row of hidden vectors
      #pragma unroll
      for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
        k_vecs[i] = (reinterpret_cast<const K_vec*>(k_ptr))[i * WARP_SIZE];
        q_vecs[i] = (reinterpret_cast<K_vec*>(q_shared_ptr))[(idx + i * WARP_SIZE) % NUM_VECS_PER_TOKEN];
      }

      float qk = scale * Qk_dot<scalar_t, NUM_THREADS_PER_TOKEN>::dot(q_vecs, k_vecs);

      if (thread_group_offset == 0) {
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // there exists a __syncthreads within this function
  qk_max = block_max<NUM_WARPS, NUM_THREADS_PER_TOKEN>(red_shared_mem, qk_max);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }

  exp_sum = block_sum<NUM_WARPS>(&red_shared_mem[NUM_WARPS], exp_sum);
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  Float_vec accs[NUM_ROUNDS_PER_TOKEN];
  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    zero(accs[i]);
  }

  V_vec zero_value;
  zero(zero_value);
  for (int block_idx = warp_idx; block_idx < num_context_blocks; block_idx += NUM_WARPS) {
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    scalar_t logit;

    #pragma unroll
    for (int idx = lane; idx < BLOCK_SIZE * NUM_VECS_PER_TOKEN; idx += WARP_STRIDE) {
      const int token_idx = block_idx * BLOCK_SIZE + idx / NUM_VECS_PER_TOKEN;
      const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride
                                     + kv_head_idx * kv_head_stride
                                     + idx * VEC_SIZE;

      V_vec v_vecs[NUM_ROUNDS_PER_TOKEN];

      #pragma unroll
      for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
        v_vecs[i] = (reinterpret_cast<const V_vec*>(v_ptr))[i * WARP_SIZE];
      }

      if (token_idx >= context_len) {
        #pragma unroll
        for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
          v_vecs[i] = zero_value;
        }
      }

      logit = CastFunctor<float, scalar_t>()(logits[token_idx]);
      #pragma unroll
      for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
        accs[i] = TernaryOpFunctor<scalar_t, V_vec, Float_vec, TernaryOpType::kFma>()(logit, v_vecs[i], accs[i]);
      }
    }
  }

  // must insert a sync since both logits and out_shared_mem occupy the same buffer space
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    block_sum<Float_vec, NUM_WARPS, NUM_THREADS_PER_TOKEN, VEC_SIZE>(out_shared_mem, accs[i]);
  }

  scalar_t* out_ptr = out + seq_idx * q_stride + head_idx * HEAD_SIZE;
  L_vec out_reg;
  #pragma unroll
  for (int i = 0; i < NUM_ROUNDS_PER_TOKEN; i++) {
    if (thread_idx < NUM_THREADS_PER_TOKEN) {
      out_reg = CastFunctor<Float_vec, L_vec>()(accs[i]);
      (reinterpret_cast<L_vec*>(out_ptr))[thread_idx + i * NUM_THREADS_PER_TOKEN] = out_reg;
    }
  }
}

#define LAUNCH_FLASH_DECODING_ATTENTION_V1(HEAD_SIZE)                                         \
  cudaFuncSetAttribute(                                                                       \
    ((void*)flash_decoding_attention_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>), \
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);                            \
  flash_decoding_attention_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>             \
                       <<<grid, block, shared_mem_size, stream>>>(                            \
    reinterpret_cast<T*>(out.data_ptr()),                                                     \
    reinterpret_cast<T*>(query.data_ptr()),                                                   \
    reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                                         \
    reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                                       \
    context_lens.data_ptr<int>(),                                                             \
    block_tables.data_ptr<int>(),                                                             \
    max_context_len,                                                                          \
    num_kv_heads,                                                                             \
    scale,                                                                                    \
    max_num_blocks_per_seq,                                                                   \
    q_stride,                                                                                 \
    kv_block_stride,                                                                          \
    kv_head_stride);

template<
  typename T,
  typename CACHE_T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void flash_decoding_attention_v1_launcher(
  torch::Tensor& out,              // [num_tokens, num_heads, head_size]
  torch::Tensor& query,            // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& value_cache,      // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& context_lens,     // [num_tokens]
  torch::Tensor& block_tables,     // [num_tokens, max_num_blocks_per_seq]
  int max_context_len,
  float scale) {
  int num_tokens = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
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
  int shared_mem_size = std::max(logits_size, outputs_size);

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
    scale);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T)                       \
  switch (block_size) {                                               \
    case 8:                                                           \
      CALL_V1_LAUNCHER(T, CACHE_T, 8);                                \
      break;                                                          \
    case 16:                                                          \
      CALL_V1_LAUNCHER(T, CACHE_T, 16);                               \
      break;                                                          \
    case 32:                                                          \
      CALL_V1_LAUNCHER(T, CACHE_T, 32);                               \
      break;                                                          \
    default:                                                          \
      AT_ERROR("block size must be 8, 16, 32");                       \
      break;                                                          \
  }

void flash_decoding_attention(
  torch::Tensor& out,             // [num_tokens, num_heads, head_size]
  torch::Tensor& query,           // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& value_cache,     // [num_blocks, num_kv_heads, block_size, head_size]
  torch::Tensor& context_lens,    // [num_tokens]
  torch::Tensor& block_tables,    // [num_tokens, max_num_blocks_per_seq]
  int block_size,
  int max_context_len,
  torch::Tensor& tmp_out,         // [num_tokens, num_heads, max_num_partitions, head_size]
  torch::Tensor& tmp_out_lse,     // [num_tokens, num_heads, max_num_partitions]
  float scale) {
  switch (query.scalar_type()) {
    case at::ScalarType::Float:
      CALL_V1_LAUNCHER_BLOCK_SIZE(float, float);
      break;
    case at::ScalarType::Half:
      CALL_V1_LAUNCHER_BLOCK_SIZE(half, half);
      break;
    case at::ScalarType::BFloat16:
      CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16, __nv_bfloat16);
      break;
    default:
      AT_ERROR("Unsupported data type: ", toString(query.scalar_type()));
  }
}


#undef LAUNCH_FLASH_DECODING_ATTENTION_V1
#undef CALL_V1_LAUNCHER
#undef CALL_V1_LAUNCHER_BLOCK_SIZE
