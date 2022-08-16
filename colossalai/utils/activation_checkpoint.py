#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.checkpoint import check_backward_validity, detach_variable

from colossalai.context.random import get_states, get_current_mode, set_seed_states, set_mode, sync_states
from .cuda import get_current_device

import weakref


def copy_to_device(obj, device):
    if torch.is_tensor(obj):
        # Notice:
        # When in no_grad context, requires_gard is False after movement
        ret = obj.to(device).detach()
        ret.requires_grad = obj.requires_grad
        return ret
    elif isinstance(obj, list):
        return [copy_to_device(i, device) for i in obj]
    elif isinstance(obj, tuple):
        return tuple([copy_to_device(v, device) for v in obj])
    elif isinstance(obj, dict):
        return {k: copy_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, activation_offload=False, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.activation_offload = activation_offload
        ctx.device = get_current_device()

        # preserve rng states
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        sync_states()
        ctx.fwd_seed_states = get_states(copy=True)
        ctx.fwd_current_mode = get_current_mode()

        if hasattr(torch, 'is_autocast_enabled'):
            ctx.had_autocast_in_fwd = torch.is_autocast_enabled()
        else:
            ctx.had_autocast_in_fwd = False

        if activation_offload:
            inputs_cuda = copy_to_device(args, ctx.device)
        else:
            inputs_cuda = args

        with torch.no_grad():
            outputs = run_function(*inputs_cuda)
        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                if activation_offload:
                    tensor_inputs.append(copy_to_device(arg, 'cpu'))
                else:
                    tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        if activation_offload:
            ctx.tensor_inputs = tensor_inputs
        else:
            ctx.save_for_backward(*tensor_inputs)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad() or when an `inputs` parameter is "
                               "passed to .backward(). Please use .backward() and do not pass its `inputs` argument.")
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices

        if ctx.activation_offload:
            tensors = ctx.tensor_inputs
        else:
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
        if ctx.activation_offload:
            tensors = copy_to_device(tensors, ctx.device)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]
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
            raise RuntimeError("none of output has requires_grad=True,"
                               " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, activation_offload, *args, use_reentrant: bool = True):
    """Checkpoint the computation while preserve the rng states, modified from Pytorch torch.utils.checkpoint.

    Args:
        function: Describe the forward pass function. It should know how to handle the input tuples.
        activation_offload: The variable to check whether we should offload activation to cpu 
        args (list): Tuple containing the parameters of the function
        use_reentrant: Bool type to check if we need to use_reentrant, if use_reentrant=False, there
        might be more flexibility for user to define there checkpoint function

    Returns:
        Output of running function with provided args.
    """
    if use_reentrant:
        return CheckpointFunction.apply(function, activation_offload, *args)
    else:
        return _checkpoint_without_reentrant(
            function,
            activation_offload,
            *args,
        )


def _checkpoint_without_reentrant(function, activation_offload=False, *args):
    # store rng_state
    fwd_cpu_state = torch.get_rng_state()
    sync_states()
    fwd_seed_states = get_states(copy=True)
    fwd_current_mode = get_current_mode()

    # check if use autocast
    if hasattr(torch, 'is_autocast_enabled'):
        has_autocast_in_fwd = torch.is_autocast_enabled()
    else:
        has_autocast_in_fwd = False

    # using WeakKeyDictionary to store all the activation the first time we call unpack
    storage: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    weak_holder_list = []

    # class for weakref.ref
    class Holder():
        pass

    # return a Holder object for later unpack process
    def pack(x):
        res = Holder()
        weak_holder_list.append(weakref.ref(res))
        return res

    # unpack hook
    def unpack(x):
        unpack_counter = 0

        # re-compute all the activation inside the function when we first call unpack
        if len(storage) == 0:

            def inner_pack(inner):
                nonlocal unpack_counter
                unpack_counter += 1

                # If the holder went out of scope, the SavedVariable is dead and so
                # the value will never be read from the storage. Skip filling it.
                if weak_holder_list[unpack_counter - 1]() is None:
                    return

                # Use detach here to ensure we don't keep the temporary autograd
                # graph created during the second forward
                storage[weak_holder_list[unpack_counter - 1]()] = inner.detach()
                return

            def inner_unpack(packed):
                raise RuntimeError("You are calling backwards on a tensor that is never exposed. Please open an issue.")

            # restore rng state
            torch.set_rng_state(fwd_cpu_state)
            for parallel_mode, state in fwd_seed_states.items():
                set_seed_states(parallel_mode, state)
            set_mode(fwd_current_mode)

            # reload arg into device if needed
            if activation_offload:
                for arg in args:
                    if torch.is_tensor(arg):
                        arg = arg.to(device=device)

            # rerun forward, the inner_pack will store all the activations in storage
            if has_autocast_in_fwd:
                with torch.enable_grad(), \
                     torch.cuda.amp.autocast(), \
                     torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args)
            else:
                with torch.enable_grad(), \
                     torch.autograd.graph.saved_tensors_hooks(inner_pack, inner_unpack):
                    _unused = function(*args)

        if x not in storage:
            raise RuntimeError("Attempt to retrieve a tensor saved by autograd multiple times without checkpoint"
                               " recomputation being triggered in between, this is not currently supported. Please"
                               " open an issue with details on your use case so that we can prioritize adding this.")

        return storage[x]

    # get device if we need to offload the activation
    if activation_offload:
        device = get_current_device()

    # run function with pack and unpack as saved_tensors_hooks
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = function(*args)

        # offload activation if needed
        if activation_offload:
            for arg in args:
                if torch.is_tensor(arg):
                    arg = arg.to(device="cpu")

    return output
