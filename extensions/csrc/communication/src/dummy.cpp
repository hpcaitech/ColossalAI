#include "dummy.hpp"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <iostream>

namespace c10d {

bool WorkDummy::isCompleted() { return true; }

bool WorkDummy::isSuccess() const { return true; }

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) { return true; }

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendDummy::BackendDummy(const c10::intrusive_ptr<::c10d::Store>& store,
                           int rank, int size)
    : Backend(rank, size), pg_nccl(store, rank, size) {
  // ::c10d::ProcessGroupNCCL
  // auto pg_options = ::c10d::ProcessGroupNCCL::Options::create();
  // auto pg_nccl = ::c10d::ProcessGroupNCCL(store, rank, size, pg_options);
}

void BackendDummy::cast_to_fp8(at::Tensor& input_tensor,
                               at::Tensor& output_tensor,
                               at::Tensor& scale_inv) {
  at::Tensor tensor_max = input_tensor.abs().max();
  at::Tensor tensor_max_new =
      torch::where(tensor_max > 0, tensor_max, at::Scalar(1));
  at::Tensor fp8_max = torch::scalar_tensor(at::Scalar(448.0));
  at::Tensor scale = fp8_max.div(tensor_max_new);
  output_tensor =
      scale.mul(input_tensor.to(torch::kFloat32)).to(at::kFloat8_e4m3fn);
  scale_inv = 1.0 / scale;
}

at::Tensor BackendDummy::cast_from_fp8(at::Tensor input_tensor,
                                       at::Tensor scale_inv,
                                       caffe2::TypeMeta dtype) {
  return scale_inv.mul(input_tensor.to(torch::kFloat32)).to(dtype);
}
// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts /* unused */) {
  return pg_nccl.allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> BackendDummy::_allgather_base(
    at::Tensor& tensor1 /* unused */, at::Tensor& tensor2 /* unused */,
    const AllgatherOptions& /* unused */ opt) {
  return pg_nccl._allgather_base(tensor1, tensor2, opt);
  // throw std::runtime_error("not supported");
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously

c10::intrusive_ptr<Work> BackendDummy::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  std::vector<int64_t> tmp_size;
  auto tensor = tensors[0];
  // int world_size = this->getSize();
  int world_size = 2;
  auto input_type = tensor.dtype();
  auto device = tensor.device();

  at::Tensor flatten_tensor = tensor.flatten();

  at::Tensor fp8_tensor;
  at::Tensor scale;
  cast_to_fp8(flatten_tensor, fp8_tensor, scale);
  fp8_tensor = fp8_tensor.view(torch::kInt8);
  auto output_tensor = torch::empty_like(fp8_tensor);

  pg_nccl.alltoall_base(output_tensor, fp8_tensor, tmp_size, tmp_size)
      ->wait(std::chrono::milliseconds(10000));

  at::Tensor scale_list = torch::zeros(
      {world_size},
      at::TensorOptions().dtype(scale.dtype()).device(scale.device()));
  pg_nccl._allgather_base(scale_list, scale)
      ->wait(std::chrono::milliseconds(10000));

  auto output_tensor_chunk = at::chunk(output_tensor, world_size);

  auto sumed_output = torch::zeros_like(output_tensor_chunk[0]).to(input_type);

  for (int rank = 0; rank < world_size; ++rank) {
    sumed_output +=
        cast_from_fp8(output_tensor_chunk[rank].view(at::kFloat8_e4m3fn),
                      scale_list[rank], input_type);
  }

  at::Tensor sumed_output_fp8;
  at::Tensor sumed_output_scale;
  cast_to_fp8(sumed_output, sumed_output_fp8, sumed_output_scale);
  sumed_output_fp8 = sumed_output_fp8.view(torch::kInt8);

  auto sumed_output_scale_list = torch::zeros(
      {world_size},
      at::TensorOptions().dtype(scale.dtype()).device(scale.device()));
  auto sumed_output_fp8_list = torch::empty_like(tensor).to(torch::kInt8);

  pg_nccl._allgather_base(sumed_output_scale_list, sumed_output_scale)
      ->wait(std::chrono::milliseconds(10000));
  pg_nccl._allgather_base(sumed_output_fp8_list, sumed_output_fp8)
      ->wait(std::chrono::milliseconds(10000));

  auto sumed_output_fp8_chunk = at::chunk(sumed_output_fp8_list, world_size);
  std::vector<at::Tensor> output;
  for (int rank = 0; rank < world_size; ++rank) {
    output.push_back(
        cast_from_fp8(sumed_output_fp8_chunk[rank].view(at::kFloat8_e4m3fn),
                      sumed_output_scale_list[rank], input_type));
  }

  tensors[0].copy_(at::cat(output).reshape(tensor.sizes()));

  auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::barrier(
    const BarrierOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */, const GatherOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce(
    std::vector<at::Tensor>& /* unused */, const ReduceOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::send(std::vector<at::Tensor>& tensors,
                                            int dstRank, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recv(std::vector<at::Tensor>& tensors,
                                            int srcRank, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Work> BackendDummy::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
  throw std::runtime_error("not supported");
}

c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& store /* unused */, int rank,
    int size, const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendDummy>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

}  // namespace c10d
