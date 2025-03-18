import pytest
import torch
from coati.distributed.utils import log_probs_from_logits

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer import dist_log_prob_1d
from colossalai.testing import rerun_if_address_is_in_use, spawn

CONFIG = dict(
    parallel=dict(data=1, pipeline=1, tensor=dict(size=2, mode="1d")),
)


def check_dist_log_prob(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost", backend="nccl")

    # prepare data
    pred = torch.randn(2, 4, 8, requires_grad=True).cuda()
    labels = torch.randint(8, (2, 4)).cuda()

    logprob = log_probs_from_logits(pred, labels)

    pred.retain_grad()
    logprob.mean().backward()

    dist_pred = pred.clone().chunk(world_size, -1)[rank].detach()
    dist_pred.requires_grad = True
    dist_logprob = dist_log_prob_1d(dist_pred, labels)

    dist_pred.retain_grad()
    dist_logprob.squeeze(-1).mean().backward()

    assert torch.allclose(
        logprob, dist_logprob.squeeze(-1), atol=1e-5
    ), f"dist cross entropy logprob is not equal to orgin logprob\n{logprob}\n{dist_logprob.squeeze(-1)}"

    pred_grad_partial = pred.grad.clone().chunk(world_size, -1)[rank].detach()
    assert torch.allclose(
        pred_grad_partial, dist_pred.grad
    ), f"dist grad is not equal to orgin grad\n{pred.grad}\n{dist_pred.grad}"


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_dist_log_prob():
    spawn(check_dist_log_prob, 2)


if __name__ == "__main__":
    test_dist_log_prob()
