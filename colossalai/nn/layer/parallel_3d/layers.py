import math
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from colossalai.communication import all_reduce, broadcast
from colossalai.constants import INPUT_GROUP_3D, INPUT_X_WEIGHT_3D, OUTPUT_GROUP_3D, OUTPUT_X_WEIGHT_3D, WEIGHT_GROUP_3D
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.nn import init as init
from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.registry import LAYERS
from colossalai.utils.checkpointing import (
    broadcast_state_dict,
    gather_tensor_parallel_state_dict,
    partition_tensor_parallel_state_dict,
)
from colossalai.utils.cuda import get_current_device

from ..utils import divide, set_tensor_parallel_attribute_by_partition, to_2tuple
from ._operation import (
    all_gather_tensor_3d,
    classifier_3d,
    layernorm_3d,
    linear_3d,
    reduce_scatter_tensor_3d,
    split_batch_3d,
    split_tensor_3d,
    vocab_parallel_classifier_3d,
)
from ._utils import get_depth_from_env, get_parallel_mode_from_env, register_async_grad_hook, swap_in_out_group


@LAYERS.register_module
class LayerNorm3D(ParallelLayer):
    r"""Layer Normalization for 3D parallelism.

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float, optional): a value added to the denominator for numerical stability, defaults to 1e-12.
        bias (bool, optional): Whether to add a bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-12, bias=True, dtype=None):

        super().__init__()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.input_x_weight_parallel_mode = get_parallel_mode_from_env(INPUT_X_WEIGHT_3D)
        self.depth = get_depth_from_env()
        self.normalized_shape = normalized_shape
        self.normalized_shape_per_partition = divide(normalized_shape, self.depth)

        self.weight = Parameter(
            torch.ones(self.normalized_shape_per_partition, device=get_current_device(), dtype=dtype))
        if bias:
            self.bias = Parameter(
                torch.zeros(self.normalized_shape_per_partition, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None
        self.variance_epsilon = eps
        self.reset_parameters()
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def reset_parameters(self) -> None:
        init.ones_()(self.weight)
        register_async_grad_hook(self.weight)
        if self.bias is not None:
            init.zeros_()(self.bias)
            register_async_grad_hook(self.bias)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight.transpose(0, 1)
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True,
                },
            )
        # broadcast in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = broadcast_state_dict(local_state, self.input_parallel_mode)
        # broadcast in weight groups
        local_state = broadcast_state_dict(local_state, self.weight_parallel_mode)

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        return layernorm_3d(
            input_,
            self.weight,
            self.bias,
            self.normalized_shape,
            self.variance_epsilon,
            self.output_parallel_mode,
            self.input_x_weight_parallel_mode,
        )


@LAYERS.register_module
class Linear3D(ParallelLayer):
    r"""Linear layer for 3D parallelism.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.output_x_weight_parallel_mode = get_parallel_mode_from_env(OUTPUT_X_WEIGHT_3D)
        self.depth = get_depth_from_env()
        self.skip_bias_add = skip_bias_add
        self.in_features_per_partition = divide(in_features, self.depth**2)
        self.out_features_per_partition = divide(out_features, self.depth)
        self.bias_features_per_partition = divide(out_features, self.depth)

        self.weight = Parameter(
            torch.empty(self.in_features_per_partition,
                        self.out_features_per_partition,
                        device=get_current_device(),
                        dtype=dtype))
        if bias:
            self.bias = Parameter(
                torch.zeros(self.bias_features_per_partition, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        swap_in_out_group()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def _sync_grad_hook(self, grad) -> Tensor:
        grad = all_reduce(grad.clone(), self.output_x_weight_parallel_mode)
        return grad

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.out_features

            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            register_async_grad_hook(self.weight)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias,
                          gpc.get_ranks_in_group(self.output_x_weight_parallel_mode)[0],
                          self.output_x_weight_parallel_mode)
                self.bias.register_hook(self._sync_grad_hook)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight.transpose(0, 1)
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
            )
        # partition in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
            )
        # partition in weight groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in weight groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
            keep_vars=keep_vars,
        )
        # gather in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
                keep_vars=keep_vars,
            )
        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            local_state[weight_key] = local_state[weight_key].transpose(0, 1)
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        output = linear_3d(
            input_,
            self.weight,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


@LAYERS.register_module
class Classifier3D(ParallelLayer):
    r"""Classifier for 3D parallelism.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.depth = get_depth_from_env()
        self.in_features_per_partition = divide(in_features, self.depth)

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.num_classes, self.in_features_per_partition, device=get_current_device(), dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.zeros(self.num_classes, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.depth)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
                broadcast(self.weight, gpc.get_ranks_in_group(self.weight_parallel_mode)[0], self.weight_parallel_mode)

            register_async_grad_hook(self.weight)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], ParallelMode.TENSOR)
                register_async_grad_hook(self.bias)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
            )
        # broadcast in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = broadcast_state_dict(local_state, self.input_parallel_mode)
        # broadcast in weight groups
        local_state = broadcast_state_dict(local_state, self.weight_parallel_mode)

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict()
        if self.has_weight:
            local_state[weight_key] = self.weight
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        return classifier_3d(
            input_,
            self.weight,
            self.bias,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )


@LAYERS.register_module
class VocabParallelClassifier3D(ParallelLayer):
    r"""Vocab parallel classifier layer for 3D parallelism.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.output_x_weight_parallel_mode = get_parallel_mode_from_env(OUTPUT_X_WEIGHT_3D)
        self.depth = get_depth_from_env()
        self.in_features_per_partition = divide(in_features, self.depth)
        self.out_features_per_partition = divide(num_classes, self.depth**2)
        self.bias_features_per_partition = divide(num_classes, self.depth)

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.out_features_per_partition,
                            self.in_features_per_partition,
                            device=get_current_device(),
                            dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = Parameter(
                torch.zeros(self.bias_features_per_partition, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        swap_in_out_group()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self) -> None:
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.depth)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            register_async_grad_hook(self.weight)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias,
                          gpc.get_ranks_in_group(self.output_x_weight_parallel_mode)[0],
                          self.output_x_weight_parallel_mode)
                register_async_grad_hook(self.bias)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
            )
        # partition in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
            )
        # partition in weight groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in weight groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
            keep_vars=keep_vars,
        )
        # gather in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
                keep_vars=keep_vars,
            )
        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        return vocab_parallel_classifier_3d(
            input_,
            self.weight,
            self.bias,
            self.input_parallel_mode,
            self.weight_parallel_mode,
            self.output_parallel_mode,
        )


@LAYERS.register_module
class PatchEmbedding3D(ParallelLayer):
    r"""2D Image to Patch Embedding.

    Args:
        img_size (int): image size.
        patch_size (int): patch size.
        in_chans (int): number of channels of input image.
        embed_size (int): size of embedding.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        flatten (bool, optional): whether to flatten output tensor, defaults to True.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.
        position_embed_initializer (:class:`typing.Callable`, optional):
            The initializer of position embedding, defaults to zeros initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 flatten: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_()):
        super().__init__()
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.input_x_weight_parallel_mode = get_parallel_mode_from_env(INPUT_X_WEIGHT_3D)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_size = embed_size
        embed_size_per_partition = embed_size // self.depth
        self.flatten = flatten

        self.weight = nn.Parameter(
            torch.empty((embed_size_per_partition, in_chans, *self.patch_size),
                        device=get_current_device(),
                        dtype=dtype))
        self.bias = nn.Parameter(torch.empty(embed_size_per_partition, device=get_current_device(), dtype=dtype))

        self.cls_token = nn.Parameter(
            torch.zeros((1, 1, embed_size_per_partition), device=get_current_device(), dtype=dtype))
        self.pos_embed = nn.Parameter(
            torch.zeros((1, self.num_patches + 1, embed_size_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)
        set_tensor_parallel_attribute_by_partition(self.bias, self.depth)
        set_tensor_parallel_attribute_by_partition(self.cls_token, self.depth)
        set_tensor_parallel_attribute_by_partition(self.pos_embed, self.depth)

    def _sync_grad_hook(self, grad) -> Tensor:
        grad = all_reduce(grad.clone(), self.input_x_weight_parallel_mode)
        return grad

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_out = self.embed_size
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            bias_initializer(self.bias, fan_in=fan_in)
            position_embed_initializer(self.pos_embed)

        src_rank = gpc.get_ranks_in_group(self.input_x_weight_parallel_mode)[0]
        broadcast(self.weight, src_rank, self.input_x_weight_parallel_mode)
        broadcast(self.bias, src_rank, self.input_x_weight_parallel_mode)
        broadcast(self.pos_embed, src_rank, self.input_x_weight_parallel_mode)

        self.weight.register_hook(self._sync_grad_hook)
        self.bias.register_hook(self._sync_grad_hook)
        self.cls_token.register_hook(self._sync_grad_hook)
        self.pos_embed.register_hook(self._sync_grad_hook)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        cls_token_key = prefix + 'cls_token'
        pos_embed_key = prefix + 'pos_embed'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias
            # cls token
            cls_token = state_dict.pop(cls_token_key, None)
            if cls_token is not None:
                local_state[cls_token_key] = cls_token
            # pos embed
            pos_embed = state_dict.pop(pos_embed_key, None)
            if pos_embed is not None:
                local_state[pos_embed_key] = pos_embed

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0,
                    cls_token_key: -1,
                    pos_embed_key: -1
                },
                partition_states={
                    weight_key: True,
                    bias_key: True,
                    cls_token_key: True,
                    pos_embed_key: True
                },
            )
        # broadcast in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = broadcast_state_dict(local_state, self.input_parallel_mode)
        # broadcast in weight groups
        local_state = broadcast_state_dict(local_state, self.weight_parallel_mode)

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        cls_token_key = prefix + 'cls_token'
        pos_embed_key = prefix + 'pos_embed'
        local_state = OrderedDict({
            weight_key: self.weight,
            bias_key: self.bias,
            cls_token_key: self.cls_token,
            pos_embed_key: self.pos_embed
        })

        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={
                    weight_key: 0,
                    bias_key: 0,
                    cls_token_key: -1,
                    pos_embed_key: -1
                },
                partition_states={
                    weight_key: True,
                    bias_key: True,
                    cls_token_key: True,
                    pos_embed_key: True
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_3d(input_,
                                input_parallel_mode=self.input_parallel_mode,
                                weight_parallel_mode=self.weight_parallel_mode)
        output = F.conv2d(input_, self.weight, self.bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)    # BCHW -> BNC

        cls_token = self.cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + self.pos_embed

        return output


@LAYERS.register_module
class Embedding3D(ParallelLayer):
    r"""Embedding for 3D parallelism.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:``torch.nn.functional.embedding`` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about initializer please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.input_x_weight_parallel_mode = get_parallel_mode_from_env(INPUT_X_WEIGHT_3D)

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = divide(embedding_dim, self.depth)
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self) -> None:
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth)

    def _sync_grad_hook(self, grad) -> Tensor:
        grad = all_reduce(grad.clone(), self.input_x_weight_parallel_mode)
        return grad

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()
        broadcast(self.weight,
                  gpc.get_ranks_in_group(self.input_x_weight_parallel_mode)[0], self.input_x_weight_parallel_mode)
        self.weight.register_hook(self._sync_grad_hook)

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={weight_key: 0},
                partition_states={weight_key: True},
            )
        # broadcast in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = broadcast_state_dict(local_state, self.input_parallel_mode)
        # broadcast in weight groups
        local_state = broadcast_state_dict(local_state, self.weight_parallel_mode)

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})

        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={weight_key: 0},
                partition_states={weight_key: True},
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_3d(input_,
                                input_parallel_mode=self.input_parallel_mode,
                                weight_parallel_mode=self.weight_parallel_mode)
        output = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        return output


@LAYERS.register_module
class VocabParallelEmbedding3D(ParallelLayer):
    r"""Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:``torch.nn.functional.embedding`` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about initializer please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.num_embeddings_per_partition = divide(self.num_embeddings, self.depth**2)
        self.embed_dim_per_partition = divide(self.embed_dim, self.depth)
        vocab_parallel_rank = gpc.get_local_rank(self.input_parallel_mode)
        self.vocab_start_index = vocab_parallel_rank * self.num_embeddings_per_partition * self.depth
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition * self.depth

        self.weight = Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim_per_partition),
                        device=get_current_device(),
                        dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.depth**3)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None and \
                self.padding_idx >= self.vocab_start_index and self.padding_idx < self.vocab_end_index:
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        # partition in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={weight_key: -1},
                partition_states={weight_key: True},
            )
        # partition in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={weight_key: 0},
                partition_states={weight_key: True},
            )
        # partition in weight groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={weight_key: 0},
            partition_states={weight_key: True},
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})

        # gather in weight groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            self.weight_parallel_mode,
            dims={weight_key: 0},
            partition_states={weight_key: True},
            keep_vars=keep_vars,
        )
        # gather in input groups
        if gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.input_parallel_mode,
                dims={weight_key: 0},
                partition_states={weight_key: True},
                keep_vars=keep_vars,
            )
        # gather in output groups
        if gpc.get_local_rank(self.input_parallel_mode) == 0 and \
                gpc.get_local_rank(self.weight_parallel_mode) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                self.output_parallel_mode,
                dims={weight_key: -1},
                partition_states={weight_key: True},
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_tensor_3d(input_, 0, self.weight_parallel_mode)

        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        weight = all_gather_tensor_3d(self.weight, 0, self.weight_parallel_mode)

        output_parallel = F.embedding(masked_input, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output_parallel[input_mask, :] = 0.
        output = reduce_scatter_tensor_3d(output_parallel, 0, self.input_parallel_mode)

        return output
