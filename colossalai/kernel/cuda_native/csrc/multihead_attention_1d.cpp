#include "multihead_attention_1d.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>

#if TORCH_VERSION_MAJOR > 1 || \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
#include <torch/csrc/distributed/c10d/Types.hpp>
#else
#include <c10d/Types.hpp>
#endif
#include <iostream>

#include "context.h"
#include "kernels.h"

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(int layer_id, int max_batch_tokens,
                                          int max_seq_len, int hidden_size,
                                          int num_heads,
                                          float attn_prob_dropout_ratio,
                                          float hidden_output_dropout_ratio,
                                          bool pre_or_postLayerNorm)
    : _layer_id(layer_id),
      _max_batch_tokens(max_batch_tokens),
      _max_seq_len(max_seq_len),
      _hidden_size(hidden_size),
      _heads(num_heads),
      _training(true),
      _pre_or_postLayerNorm(pre_or_postLayerNorm),
      _qkv_linear(
          typename FeedForward<T>::Config(3 * hidden_size, hidden_size)),
      _attn_out_linear(
          typename FeedForward<T>::Config(hidden_size, hidden_size)),
      _attn_ln(typename Normalize_Layer<T>::Config(hidden_size, false),
               _max_batch_tokens),
      _softmax(typename Softmax<T>::Config(num_heads)),
      _attn_prob_dropout(typename Dropout<T>::Config(attn_prob_dropout_ratio),
                         _max_batch_tokens * _heads * _max_seq_len),
      _attn_dropout(typename Dropout<T>::Config(hidden_output_dropout_ratio),
                    _max_batch_tokens * _hidden_size),
      _attn_scores(typename StridedBatchGemm<T>::Config(
          (T(1.0) / T(sqrt(_hidden_size / _heads))), T(0.0), CUBLAS_OP_T,
          CUBLAS_OP_N)),
      _attn_context(typename StridedBatchGemm<T>::Config(
          T(1.0), T(0.0), CUBLAS_OP_N, CUBLAS_OP_N)) {
  assert(_hidden_size % _heads == 0);
}

template <typename T>
MultiHeadAttention<T>::~MultiHeadAttention() {
  free_mem_buffer();
}

template <typename T>
void MultiHeadAttention<T>::attn_layer_fw(const T *input_ptr,
                                          const T *input_mask_ptr,
                                          T *output_ptr, T *buffer) {
  T *q_tf_ptr = _qkv_ptr;
  T *k_tf_ptr = q_tf_ptr + _batch_dim / pg_size;
  T *v_tf_ptr = k_tf_ptr + _batch_dim / pg_size;

  if (_pre_or_postLayerNorm) {
    _attn_ln.Forward(_gemmQKV_inp_ptr, input_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.reset_size(3 * _hidden_size / pg_size, _hidden_size);
  _qkv_linear.Forward(_batch_tokens, gemmQKV_inp_ptr, _attn_qkvw_ptr, buffer,
                      _cublasHandle);

  launch_bias_add_transform_20314<T>(q_tf_ptr, buffer, _attn_qkvb_ptr,
                                     _batch_size, _seq_len, 3, _heads / pg_size,
                                     _hidden_size / _heads, _stream);

  // attention scores, q*k
  _attn_scores.Forward(_batch_heads, _soft_out_ptr, k_tf_ptr, q_tf_ptr,
                       _cublasHandle);

  // Softmax + Mask
  _softmax.reset_size(_heads / pg_size);
  _softmax.Forward(_soft_out_ptr, input_mask_ptr, _batch_size, _seq_len,
                   _seq_len, _stream, true);

  // attn prob dropout.
  _attn_prob_dropout.dropout(_ctx_bufB_ptr, _soft_out_ptr,
                             _batch_heads * _seq_len * _seq_len, _stream);

  // attention context, score * v
  _attn_context.Forward(_batch_heads, buffer, v_tf_ptr, _ctx_bufB_ptr,
                        _cublasHandle);

  // [b, nh, s, ad] -> [b, s, nh, ad]
  launch_transform4d_0213<T>(_attn_o_inp_ptr, buffer, _batch_size, _seq_len,
                             _hidden_size / pg_size, _heads / pg_size, 1,
                             _stream);

  _attn_out_linear.reset_size(_hidden_size, _hidden_size / pg_size);
  _attn_out_linear.Forward(_batch_tokens, _attn_o_inp_ptr, _attn_ow_ptr,
                           output_ptr, _cublasHandle);

  // allreduce
  if (pg == c10::detail::UniqueVoidPtr() || pg->getSize() == 1) {
  } else {
    auto data_type = torch::kFloat;
    if (typeid(T) != typeid(float)) {
      data_type = torch::kHalf;
    }
    auto output_tensor = torch::from_blob(
        output_ptr, {int(_batch_size), int(_seq_len), int(_hidden_size)},
        torch::TensorOptions(torch::kCUDA).dtype(data_type));
    std::vector<torch::Tensor> allreduce_tensors = {output_tensor};
    auto work = pg->allreduce(allreduce_tensors, c10d::AllreduceOptions());
    work->wait();
  }

  _attn_dropout.bias_dropout_residual(output_ptr, output_ptr, input_ptr,
                                      _attn_ob_ptr, _batch_tokens, _hidden_size,
                                      _stream);
  if (!_pre_or_postLayerNorm) {
    // in-place ln since ln-input will not be used in post-ln mode
    _attn_ln.Forward(output_ptr, output_ptr, _attn_nw_ptr, _attn_nb_ptr,
                     _batch_tokens, _stream);
  }
}

template <typename T>
void MultiHeadAttention<T>::Forward(const T *input_ptr, const T *input_mask_ptr,
                                    T *out_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *attn_buffer = _shared_mem_ptr;  // 3 * _batch_dim

  attn_layer_fw(input_ptr, input_mask_ptr, out_ptr, attn_buffer);
}

template <typename T>
void MultiHeadAttention<T>::attn_layer_bw(const T *input_ptr,
                                          const T *input_mask_ptr,
                                          const T *output_ptr,
                                          const T *grad_output_ptr,
                                          T *grad_input_ptr, T *buffer) {
  cudaStream_t streams[2] = {_stream, _stream};

  const T *q_tf_ptr = _qkv_ptr;
  const T *k_tf_ptr = q_tf_ptr + _batch_dim / pg_size;
  const T *v_tf_ptr = k_tf_ptr + _batch_dim / pg_size;
  // batch_dim = batch_size * seq_len * hidden_size
  // buffer size: batch_dim * 3 + max(batch_dim * 3,
  //     batch_size * head_num * seq_len * seq_len)
  T *grad_residual_ptr = buffer;
  buffer += _batch_dim;

  T *grad_input_buf_ptr = buffer;  // batch_dim
  T *grad_qkv_5d_ptr = buffer;     // batch_dim * 3
  buffer += 3 * _batch_dim / pg_size;

  T *grad_qkv_4d_ptr = buffer;   // batch_dim * 3
  T *grad_softmax_ptr = buffer;  // batch_size * head_num * seq_len * seq_len
  // buffer += max(3 * _batch_dim,
  //   batch_size * head_num * seq_len * seq_len);

  if (_pre_or_postLayerNorm) {
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_output_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  } else {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_residual_ptr,
                      grad_output_ptr, nullptr, output_ptr, _attn_nw_ptr,
                      _attn_nb_ptr, _batch_tokens, streams);
    _attn_dropout.d_bias_dropout_residual(grad_input_ptr, _grad_attn_ob_ptr,
                                          grad_residual_ptr, _batch_tokens,
                                          _hidden_size, _stream);
  }

  // bw of output project
  _attn_out_linear.reset_size(_hidden_size, _hidden_size / pg_size);
  _attn_out_linear.Backward(_batch_tokens, grad_input_ptr, _attn_o_inp_ptr,
                            _attn_ow_ptr, _grad_attn_ow_ptr, _grad_attn_ob_ptr,
                            _cublasHandle, _stream, grad_input_buf_ptr, nullptr,
                            false);
  launch_transform_0213<T>(grad_input_ptr, grad_input_buf_ptr, _batch_size,
                           _seq_len, _hidden_size / pg_size, _heads / pg_size,
                           _stream);

  // bw of score * v
  _attn_context.Backward(
      _batch_heads, grad_input_ptr, v_tf_ptr, _ctx_bufB_ptr, _cublasHandle,
      grad_qkv_5d_ptr + 2 * _batch_dim / pg_size, grad_softmax_ptr);

  _attn_prob_dropout.d_dropout(grad_softmax_ptr,
                               _batch_heads * _seq_len * _seq_len, _stream);

  _softmax.reset_size(_heads / pg_size);
  _softmax.Backward(grad_softmax_ptr, _soft_out_ptr, _batch_size, _seq_len,
                    _seq_len, _stream);

  // bw of q * k
  _attn_scores.Backward(_batch_heads, grad_softmax_ptr, k_tf_ptr, q_tf_ptr,
                        _cublasHandle, grad_qkv_5d_ptr + _batch_dim / pg_size,
                        grad_qkv_5d_ptr);

  // [3, b, nh, s, ad] -> [b, s, 3, h]
  launch_transform4d_0213<T>(grad_qkv_4d_ptr, grad_qkv_5d_ptr, _batch_size,
                             _seq_len, _hidden_size / pg_size, _heads / pg_size,
                             3, _stream);

  const T *gemmQKV_inp_ptr =
      _pre_or_postLayerNorm ? _gemmQKV_inp_ptr : input_ptr;
  _qkv_linear.reset_size(3 * _hidden_size / pg_size, _hidden_size);
  _qkv_linear.Backward(_batch_tokens, grad_qkv_4d_ptr, gemmQKV_inp_ptr,
                       _attn_qkvw_ptr, _grad_attn_qkvw_ptr, _grad_attn_qkvb_ptr,
                       _cublasHandle, _stream, grad_input_buf_ptr, nullptr,
                       true);

  // allreduce
  if (pg == c10::detail::UniqueVoidPtr() || pg->getSize() == 1) {
  } else {
    auto data_type = torch::kFloat;
    if (typeid(T) != typeid(float)) {
      data_type = torch::kHalf;
    }
    auto grad_input_tensor =
        torch::from_blob(grad_input_buf_ptr,
                         {int(_batch_size), int(_seq_len), int(_hidden_size)},
                         torch::TensorOptions(torch::kCUDA).dtype(data_type));
    std::vector<torch::Tensor> allreduce_tensors = {grad_input_tensor};
    auto work = pg->allreduce(allreduce_tensors, c10d::AllreduceOptions());
    work->wait();
  }

  if (_pre_or_postLayerNorm) {
    _attn_ln.Backward(_grad_attn_nw_ptr, _grad_attn_nb_ptr, grad_input_ptr,
                      grad_input_buf_ptr, grad_output_ptr, gemmQKV_inp_ptr,
                      _attn_nw_ptr, _attn_nb_ptr, _batch_tokens, streams);
  } else {
    // FIXME later
    launch_fused_add2<T>(grad_input_ptr, grad_input_buf_ptr, grad_residual_ptr,
                         _batch_size, _seq_len, _hidden_size, _stream);
  }
}

template <typename T>
void MultiHeadAttention<T>::Backward(const T *grad_output_ptr,
                                     const T *input_ptr, const T *output_ptr,
                                     const T *input_mask_ptr,
                                     T *grad_input_ptr) {
  _stream = Context::Instance().get_stream();
  _cublasHandle = Context::Instance().get_cublashandle();
  T *buffer = _shared_mem_ptr;

  /*
  buffer size needed by attn bw:
      4 * _batch_dim + max(3 * _batch_dim,
      _batch_size * _head_num * _seq_len * _seq_len);
  */
  attn_layer_bw(input_ptr, input_mask_ptr, output_ptr, grad_output_ptr,
                grad_input_ptr, buffer);
}

template <typename T>
void MultiHeadAttention<T>::SetTrainingMode(bool training) {
  // Dropout will be skipped when not in training model.
  _attn_prob_dropout.SetTrainingMode(training);
  _attn_dropout.SetTrainingMode(training);
}

template <typename T>
T *MultiHeadAttention<T>::_shared_mem_ptr = nullptr;

template class MultiHeadAttention<float>;
template class MultiHeadAttention<__half>;

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

static std::unordered_map<int, std::shared_ptr<void>> s_multihead_attention;

template <typename T>
int create_multihead_attention(int layer_id, int max_batch_tokens,
                               int max_seq_len, int hidden_dim, int num_heads,
                               float attn_prob_dropout_ratio,
                               float hidden_dropout_ratio,
                               bool pre_or_postLayerNorm,
                               c10::intrusive_ptr<c10d::ProcessGroup> pg_) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Context::Instance().set_stream(stream);
  auto layer = std::make_shared<MultiHeadAttention<T>>(
      layer_id, max_batch_tokens, max_seq_len, hidden_dim, num_heads,
      attn_prob_dropout_ratio, hidden_dropout_ratio, pre_or_postLayerNorm);

  layer->SetPG(pg_);

  s_multihead_attention[layer_id] = layer;

  std::string dtype = (std::is_same<T, __half>::value) ? "half" : "float";

  return 0;
}

template <typename T>
std::vector<torch::Tensor> multihead_attention_fw(
    int layer_id, const torch::Tensor &input, const torch::Tensor &input_mask,
    const torch::Tensor &in_proj_weight, const torch::Tensor &in_proj_bias,
    const torch::Tensor &out_proj_weight, const torch::Tensor &out_proj_bias,
    const torch::Tensor &norm_weight, const torch::Tensor &norm_bias,
    bool training_mode, bool prelayernorm) {
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  const T *input_ptr = (const T *)input.data_ptr();
  const T *input_mask_ptr = (const T *)input_mask.data_ptr();

  auto output = torch::empty_like(input);
  T *out_ptr = (T *)output.data_ptr();

  std::shared_ptr<MultiHeadAttention<T>> layer =
      std::static_pointer_cast<MultiHeadAttention<T>>(
          s_multihead_attention[layer_id]);
  layer->set_cur_batch_shape(input.size(0), input.size(1));
  layer->SetTrainingMode(training_mode);

  layer->_attn_qkvw_ptr = (const T *)in_proj_weight.data_ptr();
  layer->_attn_qkvb_ptr = (const T *)in_proj_bias.data_ptr();
  layer->_attn_ow_ptr = (const T *)out_proj_weight.data_ptr();
  layer->_attn_ob_ptr = (const T *)out_proj_bias.data_ptr();
  layer->_attn_nw_ptr = (const T *)norm_weight.data_ptr();
  layer->_attn_nb_ptr = (const T *)norm_bias.data_ptr();

  layer->Forward(input_ptr, input_mask_ptr, out_ptr);

  return {output};
}

template <typename T>
std::vector<torch::Tensor> multihead_attention_bw(
    int layer_id, const torch::Tensor &grad_dec_output,
    const torch::Tensor &output, const torch::Tensor &input,
    const torch::Tensor &input_mask, const torch::Tensor &in_proj_weight,
    const torch::Tensor &in_proj_bias, const torch::Tensor &out_proj_weight,
    const torch::Tensor &out_proj_bias, const torch::Tensor &norm_weight,
    const torch::Tensor &norm_bias) {
  auto g_output = grad_dec_output.contiguous();
  CHECK_INPUT(g_output);
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(input_mask);

  auto grad_input = torch::empty_like(input);
  auto grad_in_proj_weight = torch::empty_like(in_proj_weight);
  auto grad_in_proj_bias = torch::empty_like(in_proj_bias);
  auto grad_out_proj_weight = torch::empty_like(out_proj_weight);
  auto grad_out_proj_bias = torch::empty_like(out_proj_bias);
  auto grad_norm_weight = torch::empty_like(norm_weight);
  auto grad_norm_bias = torch::empty_like(norm_bias);

  // inputs.
  const T *grad_dec_output_ptr = (const T *)g_output.data_ptr();
  const T *input_ptr = (const T *)input.data_ptr();
  const T *output_ptr = (const T *)output.data_ptr();
  const T *input_mask_ptr = (const T *)input_mask.data_ptr();

  // outputs.
  T *grad_input_ptr = (T *)grad_input.data_ptr();

  std::shared_ptr<MultiHeadAttention<T>> layer =
      std::static_pointer_cast<MultiHeadAttention<T>>(
          s_multihead_attention[layer_id]);
  layer->set_cur_batch_shape(g_output.size(0), g_output.size(1));

  layer->_grad_attn_qkvw_ptr = (T *)grad_in_proj_weight.data_ptr();
  layer->_grad_attn_qkvb_ptr = (T *)grad_in_proj_bias.data_ptr();
  layer->_grad_attn_ow_ptr = (T *)grad_out_proj_weight.data_ptr();
  layer->_grad_attn_ob_ptr = (T *)grad_out_proj_bias.data_ptr();
  layer->_grad_attn_nw_ptr = (T *)grad_norm_weight.data_ptr();
  layer->_grad_attn_nb_ptr = (T *)grad_norm_bias.data_ptr();

  layer->Backward(grad_dec_output_ptr, input_ptr, output_ptr, input_mask_ptr,
                  grad_input_ptr);

  return {grad_input,           grad_in_proj_weight, grad_in_proj_bias,
          grad_out_proj_weight, grad_out_proj_bias,  grad_norm_weight,
          grad_norm_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multihead_attention_fw_fp32", &multihead_attention_fw<float>,
        "Multi-head Attention forward with fp32 (CUDA)");
  m.def("multihead_attention_fw_fp16", &multihead_attention_fw<__half>,
        "Multi-head Attention forward with fp16 (CUDA)");
  m.def("multihead_attention_bw_fp32", &multihead_attention_bw<float>,
        "Multi-head Attention backward with fp32 (CUDA)");
  m.def("multihead_attention_bw_fp16", &multihead_attention_bw<__half>,
        "Multi-head Attention backward with fp16 (CUDA)");
  m.def("create_multihead_attention_fp32", &create_multihead_attention<float>,
        "Create Multi-head Attention with fp32 (CUDA)");
  m.def("create_multihead_attention_fp16", &create_multihead_attention<__half>,
        "Create Multi-head Attention with fp16 (CUDA)");
}
