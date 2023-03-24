#include <torch/extension.h>

torch::Tensor moe_dispatch_cuda_forward(int s, int ec, int h,
                                        torch::Tensor batch_tokens,
                                        torch::Tensor mask,
                                        torch::Tensor dest_idx);

torch::Tensor moe_dispatch_cuda_backward(int s, int ec, int h,
                                         torch::Tensor expert_grad,
                                         torch::Tensor mask,
                                         torch::Tensor dest_idx);

torch::Tensor moe_combine_cuda_forward(int s, int e, int c, int h,
                                       torch::Tensor expert_tokens,
                                       torch::Tensor logits, torch::Tensor mask,
                                       torch::Tensor dest_idx);

std::vector<torch::Tensor> moe_combine_cuda_backward(
    int s, int e, int c, int h, torch::Tensor tokens_grad,
    torch::Tensor expert_tokens, torch::Tensor logits, torch::Tensor mask,
    torch::Tensor dest_idx);

torch::Tensor cumsum_sub_one_in_dim0(torch::Tensor mask);

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor moe_dispatch_forward(int s, int ec, int h,
                                   torch::Tensor batch_tokens,
                                   torch::Tensor mask, torch::Tensor dest_idx) {
  CHECK_INPUT(batch_tokens);
  CHECK_CUDA(mask);
  CHECK_CUDA(dest_idx);

  return moe_dispatch_cuda_forward(s, ec, h, batch_tokens, mask, dest_idx);
}

torch::Tensor moe_dispatch_backward(int s, int ec, int h,
                                    torch::Tensor expert_grad,
                                    torch::Tensor mask,
                                    torch::Tensor dest_idx) {
  CHECK_INPUT(expert_grad);
  CHECK_CUDA(mask);
  CHECK_CUDA(dest_idx);

  return moe_dispatch_cuda_backward(s, ec, h, expert_grad, mask, dest_idx);
}

torch::Tensor moe_combine_forward(int s, int e, int c, int h,
                                  torch::Tensor expert_tokens,
                                  torch::Tensor logits, torch::Tensor mask,
                                  torch::Tensor dest_idx) {
  CHECK_INPUT(expert_tokens);
  CHECK_INPUT(logits);
  CHECK_CUDA(mask);
  CHECK_CUDA(dest_idx);

  return moe_combine_cuda_forward(s, e, c, h, expert_tokens, logits, mask,
                                  dest_idx);
}

std::vector<torch::Tensor> moe_combine_backward(int s, int e, int c, int h,
                                                torch::Tensor tokens_grad,
                                                torch::Tensor expert_tokens,
                                                torch::Tensor logits,
                                                torch::Tensor mask,
                                                torch::Tensor dest_idx) {
  CHECK_INPUT(tokens_grad);
  CHECK_INPUT(logits);
  CHECK_CUDA(mask);
  CHECK_CUDA(dest_idx);

  return moe_combine_cuda_backward(s, e, c, h, tokens_grad, expert_tokens,
                                   logits, mask, dest_idx);
}

torch::Tensor moe_cumsum(torch::Tensor mask) {
  CHECK_INPUT(mask);
  return cumsum_sub_one_in_dim0(mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cumsum_sub_one", &moe_cumsum, "Fast cumsum operation in dim0");
  m.def("dispatch_forward", &moe_dispatch_forward,
        "Forward operation in MoE dispatch function");
  m.def("dispatch_backward", &moe_dispatch_backward,
        "Backward operation in MoE dispatch function");
  m.def("combine_forward", &moe_combine_forward,
        "Combine operation in MoE combine function");
  m.def("combine_backward", &moe_combine_backward,
        "Combine operation in MoE combine function");
}
