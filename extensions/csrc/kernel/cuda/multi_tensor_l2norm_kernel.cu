// modified from
// https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_l2norm_kernel.cu
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "multi_tensor_apply.cuh"
#include "common/micros.h"

#define BLOCK_SIZE 512
#define ILP 4


template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes(
    T* x, T val, int lanes = 1,
    bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize =
      blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
      final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes_max_op(
    T* x, T val, int lanes = 1,
    bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize =
      blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid + i]));
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + 32]));
    else
      final = val;
      // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1)
      final =
          fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}

template <typename T>
__device__ __forceinline__ bool is_aligned(T *p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T *dst, T *src, int dst_offset,
                                           int src_offset) {
  typedef
      typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT *)dst)[dst_offset] = ((LT *)src)[src_offset];
}

template <typename x_t>
struct L2NormFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size, volatile int *noop_gmem, TensorListMetadata<1> &tl,
      float *output, float *output_per_tensor, bool per_tensor,
      int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = (x_t *)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be
                      // sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x;
           i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next * next;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] += next * next;
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val += vals[i];

    float final = reduce_block_into_lanes(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem =
            1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) *
                              max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

// Probably better to template, but since we are not likely to support other
// norm
template <typename x_t>
struct MaxNormFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size, volatile int *noop_gmem, TensorListMetadata<1> &tl,
      float *output, float *output_per_tensor, bool per_tensor,
      int max_chunks_per_tensor) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = (x_t *)tl.addresses[0][tensor_loc];
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];  // = {0}; // this probably works too but I want to be
                      // sure...
    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) {
      vals[i] = 0.f;
      r_x[i] = 0;
    }

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x;
           i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        // load
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float next = static_cast<float>(x[i]);
            vals[ii] = fmaxf(fabsf(vals[ii]), fabsf(next));
          }
        }
      }
    }

    float val = 0.f;
    for (int i = 0; i < ILP; i++) val = fmaxf(fabsf(val), fabsf(vals[i]));

    float final = reduce_block_into_lanes_max_op(s_vals, val);

    if (threadIdx.x == 0) {
      if (!isfinite(final))
        *noop_gmem =
            1;  // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] = fmaxf(fabsf(output[blockIdx.x]), fabsf(final));
      if (per_tensor)
        output_per_tensor[(tl.start_tensor_this_launch + tensor_loc) *
                              max_chunks_per_tensor +
                          chunk_idx] = final;
    }
  }
};

__global__ void cleanup(float *output, float *output_per_tensor, float *ret,
                        float *ret_per_tensor, bool per_tensor,
                        int max_chunks_per_tensor) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) *ret = sqrt(final);
  }

  if (per_tensor) {
    float *output_this_tensor =
        output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    float val = 0;
    for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
      val += output_this_tensor[i];

    float final = reduce_block_into_lanes(vals, val);

    if (threadIdx.x == 0) ret_per_tensor[blockIdx.x] = sqrt(final);
  }
}

__global__ void cleanup_v2(float *output, float *output_per_tensor, float *ret,
                           float *ret_per_tensor, bool per_tensor,
                           int max_chunks_per_tensor, int norm_type,
                           float alpha, float beta) {
  __shared__ float vals[512];

  if (blockIdx.x == 0) {
    float val = 0;
    if (threadIdx.x < 320) val = output[threadIdx.x];

    if (norm_type == 0) {
      float final = reduce_block_into_lanes_max_op(vals, val);
      if (threadIdx.x == 0) *ret = alpha * (*ret) + beta * final;
    } else {
      float final = reduce_block_into_lanes(vals, val);
      if (threadIdx.x == 0) *ret = sqrt(alpha * (*ret) * (*ret) + beta * final);
    }
  }

  if (per_tensor) {
    float *output_this_tensor =
        output_per_tensor + blockIdx.x * max_chunks_per_tensor;

    if (norm_type == 0) {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val = fmaxf(fabsf(val), fabsf(output_this_tensor[i]));

      float final = reduce_block_into_lanes_max_op(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] =
            alpha * ret_per_tensor[blockIdx.x] + beta * final;
    } else {
      float val = 0;
      for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x)
        val += output_this_tensor[i];

      float final = reduce_block_into_lanes(vals, val);

      if (threadIdx.x == 0)
        ret_per_tensor[blockIdx.x] = sqrt(alpha * ret_per_tensor[blockIdx.x] *
                                              ret_per_tensor[blockIdx.x] +
                                          beta * final);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  bool per_tensor =
      per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor =
          (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor =
        at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    ret_per_tensor = at::empty({0}, float_options);
  }

  DISPATCH_FLOAT_AND_HALF(
      tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_cuda",
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
          L2NormFunctor<scalar_t_0>(), output.data_ptr<float>(),
          per_tensor ? output_per_tensor.data_ptr<float>() : nullptr,
          per_tensor, max_chunks_per_tensor);)

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to
  // end. I could get rid of these by hacking the functor + multi tensor harness
  // with persistence logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<per_tensor ? ntensors : 1, 512, 0, stream>>>(
      output.data_ptr<float>(),
      per_tensor ? output_per_tensor.data_ptr<float>() : nullptr,
      ret.data_ptr<float>(),
      per_tensor ? ret_per_tensor.data_ptr<float>() : nullptr, per_tensor,
      max_chunks_per_tensor);

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

// Compute and update grad norm
// Here use a per tensor norm, and blend new norm(n) and old norm(gn) by
// L-2: gn = sqrt(a * gn^2 + b * n^2)
// L-inf: gn = a * gn + b * n
void multi_tensor_norm_out_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor out,
    const float alpha, const float beta, const int norm_type) {
  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  TORCH_CHECK(tensor_lists[0][0].device() == noop_flag.device(),
              "noop flag should be on the same device as tensors");
  // we don't need global thus uses empty here
  auto output = at::empty({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor =
        (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
    if (max_chunks_this_tensor > max_chunks_per_tensor)
      max_chunks_per_tensor = max_chunks_this_tensor;
  }

  // Although it is single write then read, still need to be zero
  // Since tailing element also participate cleanup
  output_per_tensor =
      at::zeros({ntensors * max_chunks_per_tensor}, float_options);

  if (norm_type == 0) {
    DISPATCH_FLOAT_AND_HALF(
        tensor_lists[0][0].scalar_type(), 0, "multi_tensor_maxnorm_cuda",
        multi_tensor_apply<1>(
            BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
            MaxNormFunctor<scalar_t_0>(), output.data_ptr<float>(),
            output_per_tensor.data_ptr<float>(), true, max_chunks_per_tensor);)
  } else {
    DISPATCH_FLOAT_AND_HALF(
        tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_cuda",
        multi_tensor_apply<1>(
            BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
            L2NormFunctor<scalar_t_0>(), output.data_ptr<float>(),
            output_per_tensor.data_ptr<float>(), true, max_chunks_per_tensor);)
  }
  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to
  // end. I could get rid of these by hacking the functor + multi tensor harness
  // with persistence logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());

  // Adding the following device guard since it happens sometimes that the
  // tensors are on one device and the cuda stream is on another device which
  // results in ILLEGAL MEM ACCESS error.
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup_v2<<<ntensors, 512, 0, stream>>>(
      output.data_ptr<float>(), output_per_tensor.data_ptr<float>(),
      ret.data_ptr<float>(), out.data_ptr<float>(), true, max_chunks_per_tensor,
      norm_type, alpha, beta);

  return;
}
