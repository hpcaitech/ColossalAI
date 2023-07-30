#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


class CublasHandle
{
public:
    static CublasHandle& instance()
    {
        static CublasHandle handle;
        return handle;
    }

    cublasHandle_t get() const
    {
        return handle;
    }

    CublasHandle(CublasHandle const&) = delete;
    void operator=(CublasHandle const&)  = delete;

private:
    cublasHandle_t handle;

    CublasHandle()
    {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
        {
            printf("cuBLAS initialization error: %d\n", stat);
            exit(stat);
        }
    }

    ~CublasHandle()
    {
        cublasDestroy(handle);
    }
};

class CudaStream {
public:
    // Get the singleton instance
    static CudaStream& instance() {
        static CudaStream instance;
        return instance;
    }

    // Get the cudaStream_t
    cudaStream_t get() const {
        return stream;
    }

private:
    // The cudaStream_t object
    cudaStream_t stream;

    // Private constructor and destructor
    CudaStream() {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            printf("cuda stream initialization error");
            exit(-1);
        }
    }

    ~CudaStream() {
        cudaStreamDestroy(stream);
    }

    // Delete copy and assignment constructors
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};




void dense_layer_fp32_kernel(const float *in, const float *weight, float *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = -1);

void dense_layer_fp16_kernel(const __half *in, const __half *weight, __half *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);

// void cublas_Gemm_Strided_Batched(const float *A, const float *B, float *out, const int M,
//                                  const int K, const int N, const int batch_count,
//                                  cublasOperation_t trans_A, cublasOperation_t trans_B, float alpha,
//                                  float beta, cublasHandle_t cublas_handle, cudaStream_t stream,
//                                  int cublasAlgo = -1);

void cublas_Gemm_Strided_Batched_FP16_Kernel(const __half *A, const __half *B, __half *out, const int M,
                                 const int K, const int N, const int batch_count,
                                 cublasOperation_t trans_A, cublasOperation_t trans_B,
                                 __half alpha, __half beta, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);



void dense_layer_fp32_forward(torch::Tensor& in, torch::Tensor& weight, torch::Tensor& out, int cublasAlgo) {
        const int M = in.size(0);
        const int K = in.size(1);
        const int N = weight.size(1);
        // Assumes in and weight are CUDA tensors, hence can call .data_ptr.

        cublasHandle_t handle = CublasHandle::instance().get();

        // Now you can get a cudaStream_t like this:
        cudaStream_t stream = CudaStream::instance().get();

        dense_layer_fp32_kernel(in.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), M, K, N, handle, stream, cublasAlgo);

}

void dense_layer_fp16_forward(torch::Tensor& in, torch::Tensor& weight, torch::Tensor& out, int cublasAlgo = 99) {
        const int M = in.size(0);
        const int K = in.size(1);
        const int N = weight.size(1);

        cublasHandle_t handle = CublasHandle::instance().get();

        // Now you can get a cudaStream_t like this:
        cudaStream_t stream = CudaStream::instance().get();

        in = in.contiguous();
        weight = weight.contiguous();
        out = out.contiguous();

        dense_layer_fp16_kernel(reinterpret_cast<const __half*>(in.data_ptr<at::Half>()), 
                                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), 
                                reinterpret_cast<__half*>(out.data_ptr<at::Half>()), 
                                M, K, N, handle, stream, cublasAlgo);


        
}

void batch_dense_layer_fp16_forward(torch::Tensor& in, torch::Tensor& weight, torch::Tensor& out, float alpha, float beta, int cublasAlgo = 99) {
    const int batch_count = in.size(0);
    const int M = in.size(1);
    const int K = in.size(2);
    const int N = weight.size(2);

    cublasHandle_t handle = CublasHandle::instance().get();

    // Now you can get a cudaStream_t like this:
    cudaStream_t stream = CudaStream::instance().get();

    in = in.contiguous();
    weight = weight.contiguous();
    out = out.contiguous();

    cublas_Gemm_Strided_Batched_FP16_Kernel(reinterpret_cast<const __half*>(in.data_ptr<at::Half>()), 
                                reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()), 
                                reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
                                M, K, N, batch_count,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                (__half)alpha, (__half)beta, handle, stream, cublasAlgo
                                );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_layer_fp32_forward",
        &dense_layer_fp32_forward, 
	"fp32 forward of dense layer");

  m.def("dense_layer_fp16_forward",
        &dense_layer_fp16_forward,
        "fp16 forward of dense layer."
  );

  m.def("batch_gemm_strided_fwd_fp16",
        &batch_dense_layer_fp16_forward,
        "fp16 forward of batch gemm"
  );
}