#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.checkpoint import check_backward_validity, detach_variable

from colossalai.context.random import get_states, get_current_mode, set_seed_states, set_mode, sync_states


def offload_activation_checkpoint_tensor(tensor: torch.Tensor, enable=True):
    """
    Make a cpu copy of activation checkpoint tensor
    """
    if enable and tensor.is_floating_point():
        return tensor.to('cpu')
    return tensor


def restore_activation_checkpoint_tensor(tensor: torch.Tensor, enable=True):
    """
    Move cpu activatation checkpoint data to gpu
    """
    if enable and tensor.is_floating_point():
        device_copy = tensor.to(torch.cuda.current_device())
        tensor.data = device_copy.data
    return tensor


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, cpu_offload, *args):
        check_backward_validity(args)
        ctx.cpu_offload = cpu_offload
        ctx.run_function = run_function

        # preserve rng states
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        sync_states()
        ctx.fwd_seed_states = get_states(copy=True)
        ctx.fwd_current_mode = get_current_mode()

        if hasattr(torch, 'is_autocast_enabled'):
            ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
        else:
            ctx.had_autocast_in_fwd = False

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(offload_activation_checkpoint_tensor(arg, enable=cpu_offload))
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        if cpu_offload:
            del args
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # store the current states
        bwd_cpu_rng_state = torch.get_rng_state()
        sync_states()
        bwd_seed_states = get_states(copy=True)
        bwd_current_mode = get_current_mode()

        # set the states to what it used to be
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        for parallel_mode, state in ctx.fwd_seed_states.items():
            set_seed_states(parallel_mode, state)
        set_mode(ctx.fwd_current_mode)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = restore_activation_checkpoint_tensor(tensors[i], enable=ctx.cpu_offload)

        detached_inputs = detach_variable(tuple(inputs))
        if ctx.had_autocast_in_fwd:
            with torch.enable_grad(), torch.cuda.amp.autocast():
                outputs = ctx.run_function(*detached_inputs)
        else:
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # recover the rng states
        torch.set_rng_state(bwd_cpu_rng_state)
        for parallel_mode, state in bwd_seed_states.items():
            set_seed_states(parallel_mode, state)
        set_mode(bwd_current_mode)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)

        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    '''Checkpoint the computation while preserve the rng states, modified from Pytorch torch.utils.checkpoint

    :param function: describe the forward pass function. It should know how to handle the input tuples.
    :param args: tuple containing inputs to the function
    :return: Output of running function on \*args
    '''
    cpu_offload = kwargs.pop('cpu_offload', False)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
    return CheckpointFunction.apply(function, cpu_offload, *args)
