// #include "ATen/ATen.h"
// #include "ATen/AccumulateType.h"
// #include "ATen/cuda/CUDAContext.h"

#include <torch/extension.h>

#include <cuda_runtime.h>

void transpose_pad(const float* src,
                    const int batch_size, 
                    const int max_seq_len, 
                    const int64_t* seq_len_list,
                    const int head_num,
                    const int size_per_head, 
                    float* dst);

void transpose_depad(const float* src, 
                    const int batch_size,
                    const int max_seq_len,
                    const int64_t* seq_len_list,
                    const int head_num, 
                    const int size_per_head,
                    float* dst);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FP32(x) AT_ASSERTM(x.dtype() == torch::kFloat32, "Datatype not implemented")
#define CHECK_FP32_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FP32(x)
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor transpose_pad_wrapper(torch::Tensor src,
                    int batch_size, 
                    int max_seq_len, 
                    torch::Tensor seq_len_list,
                    int head_num,
                    int size_per_head){
    CHECK_FP32_INPUT(src);
    CHECK_INPUT(seq_len_list);
    
    auto options =  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
    auto dst = torch::zeros({batch_size, head_num, max_seq_len, size_per_head}, options); 
    // dst.contiguous();

    transpose_pad(src.data_ptr<float>(), batch_size, max_seq_len, seq_len_list.data_ptr<int64_t>(), head_num, size_per_head, dst.data_ptr<float>());
    return dst;
}


// const float* src, const int batch_size,
//                     const int max_seq_len,
//                     const int64_t* seq_len_list,
//                     const int head_num, const int size_per_head,
//                     float* dst

torch::Tensor transpose_depad_wrapper(torch::Tensor src, 
                    int batch_size,
                    int sum_seq,
                    int max_seq_len,
                    torch::Tensor seq_len_list,
                    int head_num, 
                    int size_per_head){
    CHECK_FP32_INPUT(src);
    CHECK_INPUT(seq_len_list);
    // int sum_seq = 0;
    // for(int i = 0; i < batch_size; i++){
    //     sum_seq += seq_len_list[i];
    // }

    auto options =  torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false); 
    auto dst = torch::zeros({1, sum_seq, head_num, size_per_head}, options); 
    // dst.contiguous();

    transpose_depad(src.data_ptr<float>(), batch_size, max_seq_len, seq_len_list.data_ptr<int64_t>(), head_num, size_per_head, dst.data_ptr<float>());
    return dst;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose_pad_wrapper", &transpose_pad_wrapper, "Transpose and Padding");
  m.def("transpose_depad_wrapper", &transpose_depad_wrapper, "Transpose and Depadding");
}