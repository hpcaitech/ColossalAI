from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch.nn as nn
from torch import Tensor

from colossalai.lazy import LazyInitContext

from .._utils import getattr_, setattr_
from ..policies.auto_policy import get_autopolicy
from ..policies.base_policy import Policy, SubModuleReplacementDescription
from .shard_config import ShardConfig
from .utils import set_tensors_to_none

__all__ = ["ModelSharder", "shard_model"]


class ModelSharder(object):
    r"""
    Shard the original huggingface model according to the policy

    Args:
        policy (:class:`Policy`): The policy to shard the model
        model (:class:`torch.Module`): The model to shard
        shard_config: The setting of distributed model
    """

    def __init__(self, model: nn.Module, policy: Policy, shard_config: ShardConfig = None) -> None:
        self.model = model
        self.shard_config = shard_config
        self.policy = get_autopolicy(self.model) if policy is None else policy

    def shard(self) -> List[Dict[int, Tensor]]:
        r"""
        Shard the model according to the policy
        """
        self.policy.set_model(self.model)
        self.policy.set_shard_config(self.shard_config)
        self._preprocess()
        # get shared params before release unheld layers, this avoid misjudgment of shared params (None is None)
        shared_params = self.policy.get_shared_params()
        held_layers = self._release_unheld_layers()
        self._replace_module(include=held_layers)
        self._materialize()
        self._postprocess()
        return shared_params

    def _preprocess(self) -> None:
        self.model = self.policy.preprocess()

    def _postprocess(self) -> None:
        self.model = self.policy.postprocess()

    def _replace_module(self, include: Optional[Set[nn.Module]] = None) -> None:
        r"""
        Replace the module according to the policy, and replace the module one by one

        Args:
            model (:class:`torch.nn.Module`): The model to shard
        """
        module_descriptions = self.policy.module_policy()
        for layer_cls, module_description in module_descriptions.items():
            attr_replacement = module_description.attribute_replacement
            param_replacement = module_description.param_replacement
            sub_module_replacement = module_description.sub_module_replacement
            method_replacement = module_description.method_replacement
            self._recursive_replace_layer(
                self.model,
                layer_cls,
                attr_replacement,
                param_replacement,
                method_replacement,
                sub_module_replacement,
                include=include,
            )

    def _recursive_replace_layer(
        self,
        module: nn.Module,
        origin_cls: Union[str, nn.Module],
        attr_replacement: Dict[str, Any],
        param_replacement: List[Callable],
        method_replacement: Dict[str, Callable],
        sub_module_replacement: List[SubModuleReplacementDescription],
        include: Optional[Set[nn.Module]] = None,
    ) -> None:
        r"""
        Reverse the replace layer operation

        Args:
            module (torch.nn.Module): The object of layer to shard
            origin_cls (Union[str, torch.nn.Module]): The origin layer class or a string of layer class name
            attr_replacement (Dict[str, Any]): The attribute dict to modify
            param_replacement (List[Callable]): The function list to get parameter shard information in policy
            method_replacement (Dict[str, Callable]):  Key is the method name, value is the method for replacement
            sub_module_replacement ((List[SubModuleReplacementDescription]): The function list to get sub module shard information in policy
            include (Set[nn.Module], optional): The set of modules to keep on current device when pipeline parallel is enabled. Defaults to None
        """
        if (isinstance(origin_cls, str) and origin_cls == module.__class__.__name__) or (
            module.__class__ == origin_cls
        ):
            if attr_replacement is not None:
                self._replace_attr(module, attr_replacement)

            if param_replacement is not None and (include is None or module in include):
                self._replace_param(module, param_replacement)

            if method_replacement is not None:
                self._replace_method(module, method_replacement)

            if sub_module_replacement is not None:
                self._replace_sub_module(module, sub_module_replacement, include)

        for name, child in module.named_children():
            self._recursive_replace_layer(
                child,
                origin_cls,
                attr_replacement,
                param_replacement,
                method_replacement,
                sub_module_replacement,
                include=include,
            )

    def _replace_attr(
        self,
        module: nn.Module,
        attr_replacement: Dict[str, Any],
    ) -> None:
        r"""
        Replace the attribute of the layer

        Args:
            module (:class:`torch.nn.Module`): The object of layer to shard
            attr_replacement (Dict): The attribute dict to modify
        """
        for k, v in attr_replacement.items():
            setattr_(module, k, v, ignore=True)

    def _replace_param(
        self,
        module: nn.Module,
        param_replacement: List[Callable],
    ) -> None:
        r"""
        Replace the parameter of the layer

        Args:
            module (:class:`torch.nn.Module`): The object of layer to shard
            param_replacement (List[Callable]): The function list to get parameter shard information in policy
        """
        for param_func in param_replacement:
            param_func(module)

    def _replace_method(self, module: nn.Module, method_replacement: Dict[str, Callable]):
        for method_name, new_method in method_replacement.items():
            # bind the new method to the module
            bound_method = MethodType(new_method, module)
            setattr(module, method_name, bound_method)

    def _replace_sub_module(
        self,
        org_layer: nn.Module,
        sub_module_replacement: List[SubModuleReplacementDescription],
        include: Optional[Set[nn.Module]] = None,
    ) -> None:
        r"""
        Shard one layer according to the policy, the layer should be the same class as the key in policy's argument_policy return dict

        Args:
            org_layer (torch.nn.Module): The origin layer object to shard
            sub_module_replacement (List[SubModuleReplacementDescription]): The sub module replacement description list
            include (Set[nn.Module], optional): The set of modules to keep on current device when pipeline parallel is enabled. Defaults to None
        """
        for description in sub_module_replacement:
            suffix = description.suffix
            target_module = description.target_module
            kwargs = {} if description.kwargs is None else description.kwargs

            assert target_module is not None, "target_module should not be None"

            native_sub_module = getattr_(org_layer, suffix, ignore=True)
            # Skip replacement if submodule is not kept by current device when pipeline parallel is enabled.
            if (include is not None) and (native_sub_module is not None) and (native_sub_module not in include):
                continue

            assert not isinstance(
                native_sub_module, target_module
            ), f"The module with suffix {suffix} has been replaced, please check the policy"

            # if it is None and we are allowed to ignore this module
            # just skip
            if description.ignore_if_not_exist and native_sub_module is None:
                continue

            try:
                replace_layer = target_module.from_native_module(
                    native_sub_module, process_group=self.shard_config.tensor_parallel_process_group, **kwargs
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to replace {suffix} of type {native_sub_module.__class__.__qualname__}"
                    f" with {target_module.__qualname__} with the exception: {e}. "
                    "Please check your model configuration or sharding policy, you can set up an issue for us to help you as well."
                )

            setattr_(org_layer, suffix, replace_layer)

    def _get_recursive_held_layers(self, held_layers: Optional[List[nn.Module]]) -> Optional[List[nn.Module]]:
        def collect_sub_modules(module: nn.Module):
            if module is None:
                return
            recursive_held_layers.append(module)
            for name, child in module.named_children():
                collect_sub_modules(child)

        recursive_held_layers = []
        for module in held_layers:
            collect_sub_modules(module)
        return recursive_held_layers

    def _release_unheld_layers(self) -> Optional[Set[nn.Module]]:
        r"""
        Release the unheld layers in the model
        """
        if self.shard_config and self.shard_config.pipeline_stage_manager:
            held_layers = self.policy.get_held_layers()
            set_tensors_to_none(self.model, exclude=set(held_layers))
            return set(self._get_recursive_held_layers(held_layers))
        return None

    def _materialize(self) -> None:
        r"""
        Materialize the model if lazy initialization is used
        """
        LazyInitContext.materialize(self.model)
