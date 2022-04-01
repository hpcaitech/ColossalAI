from colossalai.zero.shard_utils.tensor_utils import colo_model_data_tensor_move, colo_model_data_tensor_move_inline
from colossalai.utils import free_port
from colossalai.testing import rerun_on_exception
from colossalai.zero.sharded_param import ShardedTensor
import colossalai

import torch

import torch.multiprocessing as mp


def run_tensor_move(rank):
    colossalai.launch(config={}, rank=0, world_size=1, host='localhost', port=free_port(), backend='nccl')

    src_t = torch.ones(2, 3).cuda()
    tgt_t = torch.zeros(2, 3)

    colo_model_data_tensor_move(src_t, tgt_t)
    assert (torch.sum(tgt_t) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    src_t = torch.ones(2, 3)
    tgt_t = torch.zeros(2, 3).cuda().half()
    colo_model_data_tensor_move(src_t, tgt_t)
    # the src_t has been removed
    assert (src_t.numel() == 0)
    assert (torch.sum(tgt_t) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    src_t = ShardedTensor(torch.ones(2, 3))
    tgt_t = ShardedTensor(torch.zeros(2, 3).cuda().half())
    colo_model_data_tensor_move(src_t, tgt_t)
    assert (torch.sum(tgt_t.payload) == 6.0), f"{torch.sum(tgt_t.payload)} vs. 6.0"

    assert (tgt_t.device.type == 'cuda')
    colo_model_data_tensor_move_inline(tgt_t, torch.device('cpu'))
    assert (tgt_t.device.type == 'cpu')


@rerun_on_exception(exception_type=mp.ProcessRaisedException, pattern=".*Address already in use.*")
def test_tensor_move():
    mp.spawn(run_tensor_move, nprocs=1)


if __name__ == '__main__':
    test_tensor_move()
