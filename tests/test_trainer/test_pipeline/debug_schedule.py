# referenced from Megatron and used to testify communication
import os.path as osp

import pytest
import torch
from torch.utils.data import DataLoader

from colossalai.builder import ModelInitializer, build_dataset, build_optimizer, build_loss
from colossalai.communication import p2p as p2p_communication
from colossalai.communication.utils import send_tensor_meta, recv_tensor_meta
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.initialize import initialize
from colossalai.utils import print_rank_0, get_current_device

NUM_BATCH = 128
NUM_MICRO = 6


def get_num_microbatches():
    return NUM_MICRO


def to_cuda(data):
    if isinstance(data, (tuple, list)):
        data = data[0].to(get_current_device())
    else:
        data = data.to(get_current_device())
    return data


def step_func(loss):
    def _step_func(input_tensor, model):
        output = model(input_tensor)
        if isinstance(output, (tuple, list)):
            if len(output) > 1:
                raise NotImplementedError("Multiple output!!!")
            else:
                output = output[0]
        return output, loss

    return _step_func


def forward_step(forward_step_func, data_iterator, model, input_tensor, losses_reduced):
    """Forward step for passed-in model.
    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.
    Returns output tensor."""

    if input_tensor is None:
        data, label = data_iterator.next()
        input_tensor = to_cuda(data)

    output_tensor, loss_func = forward_step_func(input_tensor, model)
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        data, label = data_iterator.next()
        label = to_cuda(label)
        output_tensor = loss_func(output_tensor, label) / get_num_microbatches()
        losses_reduced.append(output_tensor)

    return output_tensor


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.
    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.
    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # Retain the grad on the input_tensor.
    if input_tensor is not None:
        input_tensor.retain_grad()

    # Backward pass.
    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    # Collect the grad of the input_tensor.
    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad

    return input_tensor_grad


def forward_backward_pipelining_without_interleaving(forward_step_func, data_iterator,
                                                     model, optimizer, forward_only):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.
    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = \
        (gpc.get_world_size(ParallelMode.PIPELINE) -
         gpc.get_local_rank(ParallelMode.PIPELINE) - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    losses_reduced = []

    # Used for tensor meta information communication
    ft_shape = None
    bt_shape = None
    fs_checker = True

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            ft_shape = recv_tensor_meta(ft_shape)
        input_tensor = p2p_communication.recv_forward(ft_shape)
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        if not gpc.is_last_rank(ParallelMode.PIPELINE):
            bt_shape = output_tensor.shape
            fs_checker = send_tensor_meta(output_tensor, fs_checker)
        p2p_communication.send_forward(output_tensor)

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        if not gpc.is_first_rank(ParallelMode.PIPELINE):
            ft_shape = recv_tensor_meta(ft_shape)
        input_tensor = p2p_communication.recv_forward(ft_shape)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))

        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, losses_reduced)
        if forward_only:
            p2p_communication.send_forward(output_tensor)

            if not last_iteration:
                input_tensor = p2p_communication.recv_forward(ft_shape)

        else:
            output_tensor_grad = \
                p2p_communication.send_forward_recv_backward(output_tensor, bt_shape)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            if last_iteration:
                input_tensor = None
                p2p_communication.send_backward(input_tensor_grad)
            else:
                input_tensor = \
                    p2p_communication.send_backward_recv_forward(input_tensor_grad, ft_shape)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communication.recv_backward(bt_shape)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            p2p_communication.send_backward(input_tensor_grad)

    return losses_reduced


DIR_PATH = osp.dirname(osp.realpath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, '../configs/pipeline_vanilla_vit.py')


@pytest.mark.skip(reason="This is only for debugging purpose, please ignore this test")
@pytest.mark.dist
def test_schedule():
    initialize(CONFIG_PATH)

    # build model
    model = ModelInitializer(gpc.config.model, 1).model_initialize()
    print_rank_0('model is created')

    # keep the same sampler for all process
    torch.manual_seed(1331)

    dataset = build_dataset(gpc.config.data.dataset)
    dataloader = DataLoader(dataset=dataset, **gpc.config.data.dataloader)
    print_rank_0('train data is created')

    # build optimizer and loss
    optim = build_optimizer(gpc.config.optimizer, model)
    loss = build_loss(gpc.config.loss)
    print_rank_0('optim and loss is created')

    forward_backward_pipelining_without_interleaving(
        step_func(loss),
        iter(dataloader),
        model,
        optim,
        False
    )

    gpc.destroy()
    print_rank_0('training finished')


if __name__ == '__main__':
    test_schedule()
