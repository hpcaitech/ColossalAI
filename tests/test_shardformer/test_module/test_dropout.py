import pytest
import torch
import torch.nn.functional as F

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)


def check_dropout(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    from colossalai.shardformer.layer.dropout import Dropout1D

    # prepare data
    input = torch.randn(5, 4).to('cuda')
    dropout = Dropout1D(p=0.4).to('cuda')
    output_list = []
    for i in range(2):
        output = dropout(input)
        output_list.append(output)
        dist_output_list = [torch.zeros(*output.shape).to('cuda') for _ in range(world_size)]
        torch.distributed.all_gather(dist_output_list, output)
        print(dist_output_list)
        for j in range(world_size):
            for k in range(world_size):
                if j != k:
                    mask = torch.eq(dist_output_list[i], 0.0) == torch.eq(dist_output_list[j], 0.0)
                    assert torch.all(mask) == False, f"{mask}"


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dropout():
    spawn(check_dropout, 2)


if __name__ == '__main__':
    test_dropout()
