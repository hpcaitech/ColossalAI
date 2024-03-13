
#include <c10/macros/Macros.h>
#include <cuda_fp16.h>

#include <cfloat>

#include "string"

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 1>(
    c10::BFloat16 *dst, const c10::BFloat16 *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 2>(
    c10::BFloat16 *dst, const c10::BFloat16 *src) {
  *((float *)dst) = *((float *)src);
}

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 4>(
    c10::BFloat16 *dst, const c10::BFloat16 *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 8>(
    c10::BFloat16 *dst, const c10::BFloat16 *src) {
  *((float4 *)dst) = *((float4 *)src);
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 1>(c10::Half *dst,
                                                     const c10::Half *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 2>(c10::Half *dst,
                                                     const c10::Half *src) {
  *((float *)dst) = *((float *)src);
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 4>(c10::Half *dst,
                                                     const c10::Half *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 8>(c10::Half *dst,
                                                     const c10::Half *src) {
  *((float4 *)dst) = *((float4 *)src);
}

template <>
__device__ __inline__ void copy_vector<float, 1>(float *dst, const float *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<float, 2>(float *dst, const float *src) {
  *((float2 *)dst) = *((float2 *)src);
}

template <>
__device__ __inline__ void copy_vector<float, 4>(float *dst, const float *src) {
  *((float4 *)dst) = *((float4 *)src);
}

template <>
__device__ __inline__ void copy_vector<float, 8>(float *dst, const float *src) {
  // Since the maximum memory alignment length is 128 bits, we choose float4
  // here.
  *((float4 *)dst) = *((float4 *)src);
  *((float4 *)(dst + 4)) = *((float4 *)(src + 4));
}

template <typename T>
int get_vec_size(const torch::Tensor &tensor) {
  uint64_t address = reinterpret_cast<uint64_t>(tensor.data_ptr<T>());
  const int max_aligned_size = 128;
  const int dtype_size = sizeof(T) * 8;

  const int vec_size = max_aligned_size / sizeof(T) / 8;

  if (address % (dtype_size * 4) == 0) {
    return std::min(4, vec_size);
  } else if (address % (dtype_size * 2) == 0) {
    return std::min(2, vec_size);
  } else {
    return 1;
  }
}
