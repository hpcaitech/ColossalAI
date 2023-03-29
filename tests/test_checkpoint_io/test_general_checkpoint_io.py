import tempfile

import torch
from torch.optim import Adam
from torchvision.models import resnet18

from colossalai.checkpoint_io import GeneralCheckpointIO

# ========
# Note:
# 1. due to checkpoint IO can be quite slow if tested with all models, we will only test on resnet for now
# 2. we will test on both sharded and unsharded checkpoints
# 3. TODO(FrankLeeeee): implement sharded checkpoint and test it
# ========


def test_unsharded_checkpoint():
    # create a model and optimizer
    model = resnet18()
    optimizer = Adam(model.parameters(), lr=0.001)

    # create test data sample
    x = torch.randn(1, 3, 224, 224)

    # run fwd and bwd
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    # create a temp file for checkpoint
    model_ckpt_tempfile = tempfile.NamedTemporaryFile()
    optimizer_ckpt_tempfile = tempfile.NamedTemporaryFile()

    # save the model and optimizer
    ckpt_io = GeneralCheckpointIO()
    ckpt_io.save_model(model, model_ckpt_tempfile.name)
    ckpt_io.save_optimizer(optimizer, optimizer_ckpt_tempfile.name)

    # create new model
    new_model = resnet18()
    new_optimizer = Adam(new_model.parameters(), lr=0.001)

    # load the model and optimizer
    ckpt_io.load_model(new_model, model_ckpt_tempfile.name)
    ckpt_io.load_optimizer(new_optimizer, optimizer_ckpt_tempfile.name)

    # do recursive check for the optimizer state dict
    # if the value is a dict, compare its values
    # if the value is a list, comapre all elements one-by-one
    # if the value is a torch.Tensor, use torch.equal
    # otherwise use assertEqual
    def recursive_check(d1, d2):
        for k, v in d1.items():
            if isinstance(v, dict):
                recursive_check(v, d2[k])
            elif isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], torch.Tensor):
                        assert torch.equal(v[i], d2[k][i])
                    else:
                        assert v[i] == d2[k][i]
            elif isinstance(v, torch.Tensor):
                assert torch.equal(v, d2[k])
            else:
                assert v == d2[k]

    # check for model and optimizer state dict recursively
    recursive_check(model.state_dict(), new_model.state_dict())
    recursive_check(optimizer.state_dict(), new_optimizer.state_dict())
