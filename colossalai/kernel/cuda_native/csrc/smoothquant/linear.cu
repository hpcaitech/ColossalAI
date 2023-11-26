// modified from https://github.com/Guangxuan-Xiao/torch-int/blob/main/torch_int/kernels/linear.cu

#include "linear.h"
#include <cutlass/core_io.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <torch/torch.h>
torch::Tensor linear_silu_a8_w8_bfp32_ofp32(torch::Tensor input,  // INT8
                                       torch::Tensor weight, // INT8
                                       torch::Tensor bias,   // FP32
                                       float alpha,          // FP32
                                       float beta            // FP32
) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);

  using ElementOutput = float;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = float;
  using ElementInputA = int8_t; // <- data type of elements in input matrix A
  using ElementInputB = int8_t; // <- data type of elements in input matrix B

  // The code section below describes matrix layout of input and output
  // matrices. Column Major for Matrix A, Row Major for Matrix B and Row Major
  // for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

#if CUDA_ARCH  >= 800
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 64>,
      cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
	  EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif CUDA_ARCH  >= 750
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
      ElementOutput, // <- data type of output matrix
      128 / cutlass::sizeof_bits<
                ElementOutput>::value, // <- this is the number of elements per
                                       // vectorized memory access. For half
                                       // precision, it's 8 elements. This
                                       // becomes the vector width of math
                                       // instructions in epilogue too
      ElementAccumulator,              // <- data type of accumulator
      ElementComputeEpilogue // <- data type for alpha in linear combination
                             // function
      >;

  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
	EpilogueOp>;
#elif CUDA_ARCH  >= 700
  #define USE_TORCH_SILU
  using DefaultGemmCfg = cutlass::gemm::device::DefaultGemmConfiguration<
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>;
  using Gemm = cutlass::gemm::device::Gemm<
      int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::RowMajor, ElementAccumulator,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
      DefaultGemmCfg::ThreadblockShape, DefaultGemmCfg::WarpShape,
      DefaultGemmCfg::InstructionShape,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 1, ElementAccumulator, ElementComputeEpilogue>>;
#else
  #error "Unsupported cuda arch"
#endif

  auto input_size = cutlass::MatrixCoord(M, K);
  auto weight_size = cutlass::MatrixCoord(K, N);
  auto output_size = cutlass::MatrixCoord(M, N);

  auto device = input.device();
  // use the broadcasted bias as the output
  auto out = bias.to(device).view({1, -1}).repeat({M, 1});

  // constexpr int kSparse = Gemm::kSparse;
  // How many elements of A are covered per ElementE
  // constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;
  // The size of individual meta data
  // constexpr int kMetaSizeInBits = Gemm::kMetaSizeInBits;
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::TensorRef<ElementInputA, LayoutInputA> input_ref(
      input.data_ptr<ElementInputA>(), LayoutInputA::packed(input_size));
  cutlass::TensorRef<ElementInputB, LayoutInputB> weight_ref(
      weight.data_ptr<ElementInputB>(), LayoutInputB::packed(weight_size));
  cutlass::TensorRef<ElementOutput, LayoutOutput> out_ref(
      out.data_ptr<ElementOutput>(), LayoutOutput::packed(output_size));

  typename Gemm::Arguments arguments{
      problem_size, // <- problem size of matrix multiplication
      input_ref,    // <- reference to matrix A on device
      weight_ref,   // <- reference to matrix B on device
      out_ref,      // <- reference to matrix C on device
      out_ref,      // <- reference to matrix D on device
      {alpha, beta}, 1};
  Gemm gemm_op;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot run");
  }
#ifdef USE_TORCH_SILU
#undef USE_TORCH_SILU
  out = torch::silu(out);
#endif
  return out;
}
