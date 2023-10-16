#include <torch/extension.h>

#include "linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_silu_a8_w8_bfp32_ofp32", &linear_silu_a8_w8_bfp32_ofp32,
        "Linear SiLU (INT8)");
}
