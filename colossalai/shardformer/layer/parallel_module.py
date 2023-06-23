#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import itertools
from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, Module

from colossalai.tensor.d_tensor import (
    distribute_tensor,
    distribute_tensor_with_customization,
    get_device_mesh,
    get_sharding_spec,
    is_customized_distributed_tensor,
    is_distributed_tensor,
    sharded_tensor_to_param,
    to_global,
    to_global_for_customized_distributed_tensor,
)

__all__ = ['ParallelModule']


class ParallelModule(nn.Module, ABC):

    @abstractmethod
    def from_native_module(module: nn.Module,
                           process_group: Union[ProcessGroup, List[ProcessGroup]] = None) -> "ParallelModule":
        """
        Convert a native PyTorch module to a parallelized module.

        Args:
            module (nn.Module): the module to be converted.
            process_group (ProcessGroup or list[ProcessGroup]): the process group(s) to be used for communication.
                If this is a list, the process group at the ith index of the list will correspond to the process group
                in the ith axis of the device mesh. Defaults to None, which means the global process group.
        """
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                param_ = param if keep_vars else param.detach()
                if is_distributed_tensor(param_):
                    destination[prefix + name] = to_global(param_)
                elif is_customized_distributed_tensor(param_):
                    destination[prefix + name] = to_global_for_customized_distributed_tensor(param_)
                else:
                    destination[prefix + name] = param_

        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name

            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append('While copying the parameter named "{}", '
                                      'expected torch.Tensor or Tensor-like object from checkpoint but '
                                      'received {}'.format(key, type(input_param)))
                    continue

                if is_distributed_tensor(param):
                    # shard the input param
                    device_mesh = get_device_mesh(param)
                    sharding_spec = get_sharding_spec(param)
                    sharded_tensor = distribute_tensor(input_param, device_mesh, sharding_spec)
                    input_param = sharded_tensor_to_param(sharded_tensor)
                elif is_customized_distributed_tensor(param):
                    input_param = distribute_tensor_with_customization(input_param, param.shard_fn, param.gather_fn)

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'.format(key, input_param.shape, param.shape))
                    continue

                try:
                    with torch.no_grad():
                        param.copy_(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'.format(key, param.size(), input_param.size(),
                                                                           ex.args))
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]    # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)
