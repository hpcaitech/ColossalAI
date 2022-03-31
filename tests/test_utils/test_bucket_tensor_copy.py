from colossalai.utils.memory_utils.bucket_tensor_copy import BucketizedTensorCopy
from colossalai.zero.sharded_param import ShardedParamV2
from colossalai.utils import free_port
import torch
import colossalai


def test_bucket_copy():
    # init dist env
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=free_port(), backend='nccl')

    copyer = BucketizedTensorCopy(20)

    shape_list = [(2, 3), (5), (8), (12)]
    src_param_list = []
    tgt_param_list = []
    for shape in shape_list:
        # on CPU
        src_param = torch.nn.Parameter(torch.randn(shape, dtype=torch.float, device=torch.device('cpu')))
        # on GPU
        tgt_param = ShardedParamV2(torch.nn.Parameter(torch.ones(shape, dtype=torch.half, device=torch.device('cuda'))))

        src_param_list.append(src_param)
        tgt_param_list.append(tgt_param)

        copyer.copy(src_param, tgt_param)

    copyer.flush()

    for src_param, tgt_param in zip(src_param_list, tgt_param_list):
        diff = src_param.cpu().float() - tgt_param.sharded_data_tensor.payload.cpu().float()
        assert torch.allclose(src_param.cpu().float(),
                              tgt_param.sharded_data_tensor.payload.cpu().float(),
                              rtol=1e-03,
                              atol=1e-03), f"diff {diff}"


if __name__ == '__main__':
    test_bucket_copy()
