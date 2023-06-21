import pytest
import torch
import torch.nn.functional as F

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer import cross_entropy_1d
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode='1d')),)


def check_dist_crossentropy(rank, world_size, port, ignore_index):
    disable_existing_loggers()
    colossalai.launch(config=CONFIG, rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')

    # prepare data
    pred = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.randint(8, (2, 4))
    # set some label to -100 to test the ignore index
    labels[0, -1] = ignore_index

    org_pred = pred.view(-1, 8)
    org_labels = labels.view(-1)
    org_loss = F.cross_entropy(org_pred, org_labels)

    dist_pred = pred.chunk(world_size, -1)[rank]
    dist_loss = cross_entropy_1d(dist_pred.to('cuda'), labels.to('cuda'), ignore_index=ignore_index)

    assert torch.allclose(org_loss, dist_loss,
                          atol=1e-5), f"dist cross entropy loss is not equal to orgin loss\n{org_loss}\n{dist_loss}"


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_crossentropy():
    ignore_index = -100
    spawn(check_dist_crossentropy, 2, ignore_index=ignore_index)


if __name__ == '__main__':
    test_dist_crossentropy()
