#include "conversion_utils.h"
#include "inference_cuda_layers.h"
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <ratio>
#include <stdexcept>
#include <torch/extension.h>
#include <vector>
#define SHARE_MEM_SIZE (48 *  1024)
inline __device__ float relu(const float x) { return x < 0 ? 0 : x; }
inline __device__ float gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param  = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}
inline __device__ float silu(const float x)
{
    return x  / (1 + expf(-x));
}
/***
input: the input size is [b, l, m]
weight: the weight size is [m/size(TW)*2, n]
weight_scales: the weight scales size is [m/group_size, n]
weight_zeros: the weight scales size is [m/group_size, n/size(TW)*2]
bias: linear bias [n]
input_dim0: m
input_dim1: b * l
weight_dim0: n
weight_dim1: m
block_size_m:  m for one gpu thread block
block_size_n:  n for one gpu thread block
the computation block is [block_size_m, block_size_n] for one gpu thread block
group_size: the group size for gptq quant
add_bias: the linear has bias or not
***/
template <typename T, typename TW>
__global__ void gptq_gemm(T* input,
                               TW* weight,
                               T* weight_scales,
                               TW* weight_zeros,
                               T* bias,
                               T* residual,
                               T* output,
                               uint64_t input_dim0,
                               uint64_t input_dim1,
                               uint64_t weight_dim0,
                               uint64_t weight_dim1,
                               uint64_t group_size,
                               int32_t act_type,
                               bool add_bias,
                               bool add_residual,
                               bool qkv_fused,
                               uint64_t block_size_m,
                               uint64_t block_size_n)
{
    const uint32_t n_weights = sizeof(TW) * 2; // number of compressed weights in a TW.

    uint64_t block_offset  = blockIdx.x;
    uint64_t local_tid     = threadIdx.x;
    uint64_t block_tnum    = blockDim.x;
    uint64_t block_m_start = blockIdx.y * block_size_m;
    uint64_t block_m_end   = (blockIdx.y + 1) * block_size_m;
    block_m_end            = std::min(block_m_end, weight_dim1 * n_weights);

    uint64_t group_step = 32;
    group_step          = group_step - group_step % n_weights;
    group_step          = std::min(group_step, group_size);

    uint64_t group_block = group_step / n_weights;
    uint64_t table_iter  = (group_step / 2 * 256) / block_tnum;
    uint64_t col_offset  = block_size_n * block_offset;

    __shared__ float table[16][256]; // look-up table, for 32 inputs
    __shared__ float i2sum[16];
    return;

    uint64_t qkv_offset          = 0;
    uint64_t qkv_out_base_offset = col_offset;
	uint64_t bias_base_offset = col_offset;
	uint64_t split_m_size = weight_dim1 * n_weights;
    if (qkv_fused)
    {
		split_m_size        = weight_dim1 * n_weights / 3;
        qkv_offset          = block_m_start / split_m_size;
        qkv_out_base_offset = qkv_offset * input_dim1 * weight_dim0  + col_offset;
		bias_base_offset = qkv_offset * weight_dim0 + col_offset;
    }

    float tmp_w_res     = conversion::to<float>(0.0);
    float tmp_z_res     = conversion::to<float>(0.0);
    float tmp_final_res = conversion::to<float>(0.0);
    float tmp_weight_scales;
    float tmp_weight_zero;

    uint64_t current_group_size = group_size;
    uint64_t scale_dim1_ind     = block_m_start / group_size;

    for (uint64_t i = block_m_start; i < block_m_end; i += current_group_size)
    {
        if (i + current_group_size > block_m_end)
            current_group_size = block_m_end - i;

        // // index of weight scale
        // uint64_t dind = (i / group_size) * weight_dim0 + col_offset + local_tid;
        // int32_t i_zero =
        //     ((weight_zeros[dind / n_weights] >> (((col_offset + local_tid) & 0xf) * 4)) & 0xf) + 1;

        // tmp_weight_scales = conversion::to<float>(weight_scales[dind]);
        // tmp_weight_zero   = conversion::to<float>(i_zero);
        // if (i + current_group_size > block_m_end)
        //     current_group_size = block_m_end - i;

        // index of weight scale
        uint64_t scale_index =
            scale_dim1_ind * weight_dim0 + col_offset + local_tid;
        // 4 is 4bits weight. 0xf is mask for 4 bits weight. 1 is for gptq algorithm.
        int32_t i_zero = ((weight_zeros[scale_index / n_weights] >>
                           ((scale_index & 0xf) * 4)) &
                          0xf) +
                         1;

        tmp_weight_scales = conversion::to<float>(weight_scales[scale_index]);
        tmp_weight_zero   = conversion::to<float>(i_zero);
        scale_dim1_ind += 1;
        for (uint64_t j = 0; j < current_group_size; j += group_step)
        {

// compute lookup table
#pragma unroll
            for (uint64_t k = 0; k < table_iter; k++)
            {

                // uint64_t table_id     = k * block_tnum + local_tid;
                // uint64_t dind         = table_id & 0xff;
                // uint64_t tid          = table_id >> 8;
                // uint64_t input_offset = i + j + tid * 2;

                uint64_t table_id     = k * block_tnum + local_tid;
                uint64_t weight_id    = table_id & 0xff;
                uint64_t input_id     = table_id >> 8;
                // 2 is number of inputs  for one table elements. 
                uint64_t input_offset = (i + j + input_id * 2) % split_m_size;

                // float i1, i2;

                // float i1 = relu(conversion::to<float>(input[input_offset]));
                // float i2 = relu(conversion::to<float>(input[input_offset + 1]));
                float i1 = (conversion::to<float>(input[input_offset]));
                float i2 = (conversion::to<float>(input[input_offset + 1]));

                i2sum[input_id] = i1 + i2;

                int32_t iw1 = weight_id & 0xf;
                int32_t iw2 = weight_id >> 4;

                float w1 = conversion::to<float>(iw1);
                float w2 = conversion::to<float>(iw2);

                table[input_id][weight_id] = w1 * i1 + w2 * i2;
            }
            __syncthreads();
#pragma unroll
            for (uint64_t k = 0; k < group_block; k++)
            {

                uint64_t base_weight_offset = ((i + j) / n_weights + k) * weight_dim0;
                uint64_t dind = base_weight_offset + col_offset + local_tid;

                TW w = weight[dind];

#pragma unroll
                for (uint64_t z = 0; z < n_weights / 2; z++)
                {
                    uint32_t k1 = k * n_weights / 2 + z;
                    TW w1   = (w >> (z * 8)) & 0xff;

                    tmp_w_res += table[k1][w1];
                    tmp_z_res += i2sum[k1];
                }
            }
        }

        tmp_final_res +=
            (tmp_w_res - tmp_z_res * tmp_weight_zero) * tmp_weight_scales;
        tmp_w_res = conversion::to<float>(0.0);
        tmp_z_res = conversion::to<float>(0.0);
    }


    if(col_offset + local_tid < input_dim0 * input_dim1)
    {
        uint64_t bias_offset = bias_base_offset + local_tid;
        float bias_v        = 0;
        float residual_v  = 0;
        if (add_bias && blockIdx.y == 0)
        {
            bias_v = conversion::to<float>(bias[bias_offset]);
            tmp_final_res += bias_v;
        }
        uint64_t dind  = qkv_out_base_offset + local_tid;
        if(act_type == 1)
        {
            tmp_final_res = relu(tmp_final_res);
        }
        else if(act_type == 2)
        {
            tmp_final_res = gelu(tmp_final_res);
        }
        else if(act_type == 3)
        {
            tmp_final_res = silu(tmp_final_res);
        }

        if (add_residual && blockIdx.y == 0){
            residual_v = conversion::to<float>(residual[dind]);
            tmp_final_res += residual_v;
        }
        T tmp_res = conversion::to<T>(tmp_final_res);
        // float *o = (float*)output;
        atomicAdd(&output[dind], tmp_res);
        // atomicAdd(&output[dind], tmp_final_res);

    }  
}



template <typename T, typename TW>
at::Tensor gptq_act_linear_layer(at::Tensor& input,
                                 at::Tensor& weight,
                                 at::Tensor& weight_scales,
                                 at::Tensor& weight_zeros,
                                 at::Tensor& bias,
                                 at::Tensor& residual,
                                 int64_t group_size,
                                 int32_t act_type,
                                 int32_t add_bias,
                                 int32_t add_residual,
                                 int32_t qkv_fused,
                                 uint64_t block_size_x,
                                 uint64_t block_size_y)
{

    uint64_t input_dim0 = input.sizes()[2];
    uint64_t input_dim1 = input.sizes()[0] * input.sizes()[1];

    uint64_t weight_dim0 = weight.sizes()[1];
    uint64_t weight_dim1 = weight.sizes()[0];


    auto options =
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

    std::vector<int64_t> out_shape;
    if (qkv_fused)
        out_shape.push_back(3);
    out_shape.push_back(input.sizes()[0]);
    out_shape.push_back(input.sizes()[1]);
    out_shape.push_back(weight.sizes()[1]);

    at::Tensor output = at::zeros(out_shape, options);

    T* input_ptr         = (T*)input.data_ptr();
    TW* weight_ptr       = (TW*)weight.data_ptr();
    T* weight_scales_ptr = (T*)weight_scales.data_ptr();
    TW* weight_zeros_ptr = (TW*)weight_zeros.data_ptr();
    T* bias_ptr          = (T*)bias.data_ptr();
    T* output_ptr        = (T*)output.data_ptr();
    T* residual_ptr      = (T*)residual.data_ptr();
    // at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
// #define BENCHMARK
#ifdef BENCHMARK
    uint32_t block_xs[] = {128, 256, 512};
    uint32_t block_ys[] = {128, 256, 512, 1024};

    for (uint32_t i = 0; i < 3; i++)
    {
        for (uint32_t j = 0; j < 4; j++)
        {

            block_size_x     = block_xs[i];
            block_size_y     = block_ys[j];
            uint32_t warm_up = 2;
            uint32_t bench   = 5;
            auto start       = std::chrono::high_resolution_clock::now();
            auto end         = std::chrono::high_resolution_clock::now();
            for (uint32_t k = 0; k < warm_up + bench; k++)
            {

                if (k == warm_up)
                    start = std::chrono::high_resolution_clock::now();

#endif
                uint64_t block_size_m = block_size_y;
                uint64_t block_size_n = block_size_x;

                uint64_t block_tnum = block_size_x;

                // printf("block size m %d %d\n", weight_dim1, weight_dim0);
                // printf("block size m %d %d\n", input_dim1, input_dim0);

                if (input_dim1 == 1)
                {

                    dim3 block_dim(block_tnum, 1, 1);
                    dim3 grid_dim(weight_dim0 / block_tnum,
                                  (weight_dim1 * sizeof(TW) * 2 + block_size_y - 1) / block_size_y,
                                  1);
                // printf("block size m %d %d\n", weight_dim1, weight_dim0);
                // printf("block size m %d %d\n", input_dim1, input_dim0);
                // printf("block size m %d %d\n", block_tnum, weight_dim0 / block_tnum);
                // printf("block size m %d %d\n", weight_dim1 * sizeof(TW) * 2 / block_size_y, input_dim0);                
                    gptq_gemm<T, TW>
                        <<<grid_dim, block_dim, 0, stream>>>(input_ptr,
                                                    weight_ptr,
                                                    weight_scales_ptr,
                                                    weight_zeros_ptr,
                                                    bias_ptr,
                                                    residual_ptr,
                                                    output_ptr,
                                                    input_dim0,
                                                    input_dim1,
                                                    weight_dim0,
                                                    weight_dim1,
                                                    group_size,
                                                    act_type,
                                                    add_bias,
                                                    add_residual,
                                                    qkv_fused,
                                                    block_size_m,
                                                    block_size_n);
                }
                else
                {
                    printf("cuda kernel not support batch * seq_len > 1\n");
                }

#ifdef BENCHMARK
            }
            end = std::chrono::high_resolution_clock::now();
            double sec =
                (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             end - start)
                             .count()) /
                1e9 / 5;

            printf("block x: %d, block y: %d, %.8f\n",
                   block_size_x,
                   block_size_y,
                   sec);
        }
    }
#endif
    // float *o = (float*)output_ptr;
    // for(int i = 0; i < 64; i ++){
    //     printf("%f ", o[i]);
    // }
    // printf("\n");
    return output;
}

#define INSTANTIATE_ACT_GPTQ_LINEAR(T, TW)                                     \
    template at::Tensor gptq_act_linear_layer<T, TW>(                          \
        at::Tensor & input,                                                    \
        at::Tensor & weight,                                                   \
        at::Tensor & weight_scales,                                            \
        at::Tensor & weight_zeros,                                             \
        at::Tensor & bias,                                                     \
        at::Tensor & residual,                                                  \
        int64_t group_size,                                                    \
        int32_t act_type,                                                       \
        int32_t add_bias,                                                       \
        int32_t add_residual,                                                       \
        int32_t qkv_fused,                                                       \
        uint64_t block_size_x,                                                 \
        uint64_t block_size_y);

// INSTANTIATE_ACT_GPTQ_LINEAR(float, uint64_t)
INSTANTIATE_ACT_GPTQ_LINEAR(__half, uint64_t)
// INSTANTIATE_ACT_GPTQ_LINEAR(float, uint32_t)
INSTANTIATE_ACT_GPTQ_LINEAR(__half, uint32_t)
// INSTANTIATE_ACT_GPTQ_LINEAR(float, uint8_t)
INSTANTIATE_ACT_GPTQ_LINEAR(__half, uint8_t)
