#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

class ZeroCommMgr {
 public:
  cudaStream_t cuda_stream;
  ncclComm_t nccl_comm;

  ZeroCommMgr(const ncclComm_t &comm_) {
    CUDACHECK(cudaStreamCreate(&cuda_stream));
    nccl_comm = comm_;
  }
};

ZeroCommMgr *GMGR = nullptr;

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup : public c10d::ProcessGroupNCCL {
 public:
  ncclComm_t getcomm(at::Device dev) {
    ncclUniqueId ncclID;
    int rank = getRank();
    if (rank == 0) {
      ncclGetUniqueId(&ncclID);
    }

    broadcastUniqueNCCLID(&ncclID, c10d::OpType::SEND, "colossal_zero_comm",
                          rank);

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, getSize(), ncclID, rank));
    return comm;
  }
};

int create_zero_comm(c10d::ProcessGroupNCCL &pg, at::Device dev) {
  auto *hack_group = reinterpret_cast<HackNCCLGroup *>(&pg);
  GMGR = new ZeroCommMgr(hack_group->getcomm(dev));
  assert(GMGR->nccl_comm != 0);
  return 0;
}
#endif

template <typename scalar_t>
void colo_all_gather_impl(scalar_t *recvbuff, int rank, int sendcount,
                          ncclDataType_t data_type) {
  scalar_t *sendbuff = recvbuff + (rank * sendcount);
  NCCLCHECK(ncclAllGather(sendbuff, recvbuff, sendcount, data_type,
                          GMGR->nccl_comm, GMGR->cuda_stream));
  CUDACHECK(cudaStreamSynchronize(GMGR->cuda_stream));
}

int colo_all_gather(torch::Tensor &input_tensor, int rank, int world_size) {
  CHECK_INPUT(input_tensor);

  auto total_size = input_tensor.numel();
  assert(total_size % world_size == 0);
  auto sendcount = total_size / world_size;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input_tensor.scalar_type(), "colo_all_gather", ([&] {
        colo_all_gather_impl<scalar_t>(
            input_tensor.data_ptr<scalar_t>(), rank, sendcount,
            input_tensor.scalar_type() == at::ScalarType::Half ? ncclHalf
                                                               : ncclFloat);
      }));

  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef USE_C10D_NCCL
  m.def("create_comm", &create_zero_comm,
        "Create the communication environment for Colossal Zero");
#endif
  m.def("inplace_all_gather", &colo_all_gather,
        "All gather operation used in Colossal Zero");
}
