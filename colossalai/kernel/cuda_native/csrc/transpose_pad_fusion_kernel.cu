// For eliminating redundant computation.
// transpose and padding/depadding fusion to reduce the memory move.

#include <cuda_runtime.h>

__global__ void transpose_depad_kernel(const float* src, const int batch_size,
                                   const int seq_len,
                                   const int64_t* seq_lens,
                                   const int head_num, const int size_per_head,
                                   float* dst){
  
  int idx = threadIdx.x;  
  int batch_index = blockIdx.x / (head_num * seq_len);
  int head_index = (blockIdx.x % (head_num * seq_len)) / seq_len;
  int seq_index = blockIdx.x % seq_len;

  if (seq_index >= seq_lens[batch_index]) {
    return;
  }
  
  // to know the start place of each batch
  int64_t sum_len = 0;
  for (size_t i = 0; i < batch_index; ++i) {
    sum_len += seq_lens[i];
  }
  while (idx < size_per_head) {
    // set the invalid elements to 0.
    dst[(sum_len + seq_index) * (head_num * size_per_head) +
        head_index * size_per_head + idx] = src[blockIdx.x * size_per_head + idx];
    idx += blockDim.x;
  }
}

void transpose_depad(const float* src, const int batch_size,
                    const int seq_len,
                    const int64_t* seq_lens,
                    const int head_num, const int size_per_head,
                    float* dst)
{
    dim3 dimGrid(batch_size * head_num * seq_len);
    dim3 dimBlock(size_per_head);

    transpose_depad_kernel<<<dimGrid, dimBlock>>>(src, batch_size, seq_len, seq_lens, head_num, size_per_head, dst);
}



__global__ void transpose_pad_kernel(
    const float* src, const int batch_size, const int seq_len, 
    const int64_t* seq_lens, const int head_num,
    const int size_per_head, float* dst) {

  int idx = threadIdx.x;
  int batch_index = blockIdx.x / (head_num * seq_len);
  int head_index = (blockIdx.x % (head_num * seq_len)) / seq_len;
  int seq_index = blockIdx.x % seq_len;

  // to know the start place of each batch
  int64_t sum_len = 0;
  for (size_t i = 0; i < batch_index; ++i) {
    sum_len += seq_lens[i];
  }
  while (idx < size_per_head) {
    if (seq_index >= seq_lens[batch_index]) {
      dst[blockIdx.x * size_per_head + idx] = 0.f;
    } else {
      dst[blockIdx.x * size_per_head + idx] =
          src[(sum_len + seq_index) * (head_num * size_per_head) +
              head_index * size_per_head + idx]; 
    }
    idx += blockDim.x;
  }
}

void transpose_pad(const float* src,
    const int batch_size, 
    const int seq_len,
    const int64_t* seq_lens, 
    const int head_num,
    const int size_per_head, 
    float* dst)
{

    dim3 dimGrid(batch_size * head_num * seq_len);
    dim3 dimBlock(size_per_head);

    transpose_pad_kernel<<<dimGrid, dimBlock>>>(src, batch_size, seq_len, seq_lens, head_num, size_per_head, dst);
}