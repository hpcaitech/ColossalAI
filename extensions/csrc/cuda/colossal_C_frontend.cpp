// modified from
// https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_adam.cu
#include <torch/extension.h>

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists,
                             float scale);

void multi_tensor_sgd_cuda(int chunk_size, at::Tensor noop_flag,
                           std::vector<std::vector<at::Tensor>> tensor_lists,
                           float wd, float momentum, float dampening, float lr,
                           bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1,
                            const float beta2, const float epsilon,
                            const int step, const int mode,
                            const int bias_correction, const float weight_decay,
                            const float div_scale);

void multi_tensor_lamb_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1,
                            const float beta2, const float epsilon,
                            const int step, const int bias_correction,
                            const float weight_decay, const int grad_averaging,
                            const int mode, at::Tensor global_grad_norm,
                            const float max_grad_norm,
                            at::optional<bool> use_nvlamb_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_lamb", &multi_tensor_lamb_cuda,
        "Computes and apply update for LAMB optimizer");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
}
