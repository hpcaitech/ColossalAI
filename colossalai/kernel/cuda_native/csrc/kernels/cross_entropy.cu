#include "block_reduce.h"
#include "cuda_util.h"
#include "kernels.h"
#include "ls_cub.cuh"

ls::cub::CachingDeviceAllocator g_allocator(true);

template <typename T>
__global__ void ls_cross_entropy_fw_kernel(
    const T *__restrict__ inputs, const int *__restrict__ targets,
    float *__restrict__ outputs, float *__restrict__ nll_loss_outputs,
    const int padding_idx, const float epsilon, const int vocab_size) {
  /* step1: compute each thread's max_logit and sum_exp_logit, store in
   * max_input, sum_exp_logit */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float max_input[1] = {REDUCE_FLOAT_INF_NEG};
  float sum_logits[2] = {0.f, 0.f};  // logit and logit exp
  int target_tid = targets[blockIdx.x];

  if (target_tid == padding_idx) {
    if (threadIdx.x == 0) {
      nll_loss_outputs[blockIdx.x] = 0.f;
      outputs[blockIdx.x] = 0.f;
    }
    return;
  }

  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    max_input[0] = fmaxf(max_input[0], static_cast<float>(inputs[i]));
  }
  blockReduce<ReduceType::kMax, 1>(max_input);
  __shared__ float s_max_input;
  if (threadIdx.x == 0) {
    s_max_input = max_input[0];
  }
  __syncthreads();

  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float logit = static_cast<float>(inputs[i]) - s_max_input;
    sum_logits[0] += logit;
    sum_logits[1] += expf(logit);
  }

  blockReduce<ReduceType::kSum, 2>(sum_logits);
  __shared__ float s_sum_logit;
  __shared__ float s_sum_exp;
  if (threadIdx.x == 0) {
    s_sum_logit = sum_logits[0];
    s_sum_exp = sum_logits[1];
  }
  __syncthreads();

  float eps_i = epsilon / (vocab_size - 1);
  if (threadIdx.x == 0) {
    // neg_log_prob = log(sum(exp(x - x_max))) - (x - x_max)
    float nll_loss = logf(s_sum_exp) -
                     static_cast<float>(inputs[block_start + target_tid]) +
                     s_max_input;
    nll_loss_outputs[blockIdx.x] = nll_loss;
    float sum_nll_loss = vocab_size * logf(s_sum_exp) - s_sum_logit;
    outputs[blockIdx.x] =
        (1.f - epsilon - eps_i) * nll_loss + eps_i * sum_nll_loss;
  }
}

template <typename T>
__global__ void ls_cross_entropy_bw_kernel(
    const float *__restrict__ grad_outputs, const T *__restrict__ inputs,
    const int *__restrict__ targets, T *__restrict__ grad_inputs,
    const int padding_idx, const float epsilon, const int vocab_size) {
  /* step1: compute each thread's max_logit and sum_exp_logit, store in
   * max_input, sum_exp_logit */
  const int block_start = blockIdx.x * vocab_size;
  const int left_idx = block_start + threadIdx.x;
  const int right_idx = (blockIdx.x + 1) * vocab_size;
  float max_input[1] = {REDUCE_FLOAT_INF_NEG};
  float sum_logits[1] = {0.f};
  const float grad_out = static_cast<float>(grad_outputs[0]);
  int target_tid = targets[blockIdx.x];

  if (target_tid == padding_idx) {
    for (int i = left_idx; i < right_idx; i += blockDim.x) {
      grad_inputs[i] = 0.f;
    }
    return;
  }

  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    max_input[0] = fmaxf(max_input[0], static_cast<float>(inputs[i]));
  }
  blockReduce<ReduceType::kMax, 1>(max_input);
  __shared__ float s_max_input;
  if (threadIdx.x == 0) {
    s_max_input = max_input[0];
  }
  __syncthreads();

  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float logit = static_cast<float>(inputs[i]) - s_max_input;
    sum_logits[0] += expf(logit);
  }

  blockReduce<ReduceType::kSum, 1>(sum_logits);
  __shared__ float s_sum_exp;
  if (threadIdx.x == 0) {
    s_sum_exp = sum_logits[0];
  }
  __syncthreads();

  float eps_i = epsilon / (vocab_size - 1);
  float nll_weight = 1.0 - epsilon - eps_i;

  for (int i = left_idx; i < right_idx; i += blockDim.x) {
    float prob = expf(static_cast<float>(inputs[i]) - s_max_input) / s_sum_exp;
    float grad = 0;
    grad += (vocab_size * prob - 1) * eps_i;
    grad += prob * nll_weight;
    if ((i - block_start) == target_tid) {
      grad -= nll_weight;
    }
    grad_inputs[i] = grad_out * grad;
  }
}

template <typename T>
void launch_cross_entropy_fw(const T *inputs_ptr, const int *targets_ptr,
                             float *outputs_ptr, float *nll_loss_ptr,
                             float *loss_buffer, const int padding_idx,
                             const float epsilon, const int batch_size,
                             const int seq_len, const int vocab_size,
                             cudaStream_t stream) {
  int grid_dim = batch_size * seq_len;
  float *nll_loss_buffer = loss_buffer + grid_dim;
  ls_cross_entropy_fw_kernel<<<grid_dim, MAX_THREADS, 0, stream>>>(
      inputs_ptr, targets_ptr, loss_buffer, nll_loss_buffer, padding_idx,
      epsilon, vocab_size);

  int num_items = grid_dim;
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CHECK_GPU_ERROR(ls::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                             loss_buffer, outputs_ptr,
                                             num_items, stream));
  CHECK_GPU_ERROR(
      g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
  CHECK_GPU_ERROR(ls::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                             loss_buffer, outputs_ptr,
                                             num_items, stream));
  CHECK_GPU_ERROR(ls::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                             nll_loss_buffer, nll_loss_ptr,
                                             num_items, stream));
  CHECK_GPU_ERROR(g_allocator.DeviceFree(d_temp_storage));
}

template void launch_cross_entropy_fw<float>(
    const float *inputs_ptr, const int *targets_ptr, float *outputs_ptr,
    float *nll_loss_ptr, float *loss_buffer, const int padding_idx,
    const float epsilon, const int batch_size, const int seq_len,
    const int vocab_size, cudaStream_t stream);

template void launch_cross_entropy_fw<__half>(
    const __half *inputs_ptr, const int *targets_ptr, float *outputs_ptr,
    float *nll_loss_ptr, float *loss_buffer, const int padding_idx,
    const float epsilon, const int batch_size, const int seq_len,
    const int vocab_size, cudaStream_t stream);

template <typename T>
void launch_cross_entropy_bw(const float *grad_outputs_ptr, const T *inputs_ptr,
                             const int *targets_ptr, T *grad_inputs_ptr,
                             const int padding_idx, const float epsilon,
                             const int batch_size, const int seq_len,
                             const int vocab_size, cudaStream_t stream) {
  int grid_dim = batch_size * seq_len;
  ls_cross_entropy_bw_kernel<<<grid_dim, MAX_THREADS, 0, stream>>>(
      grad_outputs_ptr, inputs_ptr, targets_ptr, grad_inputs_ptr, padding_idx,
      epsilon, vocab_size);
}

template void launch_cross_entropy_bw<float>(
    const float *grad_outputs_ptr, const float *inputs_ptr,
    const int *targets_ptr, float *grad_inputs_ptr, const int padding_idx,
    const float epsilon, const int batch_size, const int seq_len,
    const int vocab_size, cudaStream_t stream);

template void launch_cross_entropy_bw<__half>(
    const float *grad_outputs_ptr, const __half *inputs_ptr,
    const int *targets_ptr, __half *grad_inputs_ptr, const int padding_idx,
    const float epsilon, const int batch_size, const int seq_len,
    const int vocab_size, cudaStream_t stream);
