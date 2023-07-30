#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

void dense_layer_fp32_kernel(const float *in, const float *weight, float *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weight,
                                CUDA_R_32F, N, in, CUDA_R_32F, K, &beta, out, CUDA_R_32F, N,
                                CUDA_R_32F, static_cast<cublasGemmAlgo_t>(cublasAlgo));
}

void dense_layer_fp16_kernel(const __half *in, const __half *weight, __half *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo) {
  const __half alpha = (__half)1.0f, beta = (__half)0.0f;
  cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, weight,
                                CUDA_R_16F, N, in, CUDA_R_16F, K, &beta, out, CUDA_R_16F, N,
                                CUDA_R_16F, static_cast<cublasGemmAlgo_t>(cublasAlgo));
}


void cublas_Gemm_Strided_Batched_FP16_Kernel(const __half *A, const __half *B, __half *out, const int M,
                                 const int K, const int N, const int batch_count,
                                 cublasOperation_t trans_A, cublasOperation_t trans_B,
                                 __half alpha, __half beta, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo) {
  const int lda = (trans_A == CUBLAS_OP_N) ? K : M;
  const int ldb = (trans_B == CUBLAS_OP_N) ? N : K;
  

  cublasGemmStridedBatchedEx(
      cublas_handle, trans_B, trans_A, N, M, K, &alpha, B, CUDA_R_16F, ldb, K * N, A, CUDA_R_16F,
      lda, M * K, &beta, out, CUDA_R_16F, N, M * N, batch_count, CUDA_R_16F,
      static_cast<cublasGemmAlgo_t>(cublasAlgo));
}
