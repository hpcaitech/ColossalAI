from typing import Any, Callable, Dict, List, Union

import torch.nn as nn

from .._utils import getattr_, setattr_
from ..policies.autopolicy import get_autopolicy
from ..policies.basepolicy import Policy, SubModuleReplacementDescription
from .shard_config import ShardConfig

__all__ = ['ModelSharder', 'shard_model']


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
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.shard_config = shard_config

    def shard(self) -> None:
        r"""
        Shard the model according to the policy
        """
        self.policy.set_model(self.model)
        self.policy.set_shard_config(self.shard_config)
        self._preprocess()
        self._replace_module()
        self._postprocess()

    def _preprocess(self) -> None:
        self.model = self.policy.preprocess()

    def _postprocess(self) -> None:
        self.model = self.policy.postprocess()

    def _replace_module(self,) -> None:
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
            self._recursive_replace_layer(self.model, layer_cls, attr_replacement, param_replacement,
                                          method_replacement, sub_module_replacement)

    def _recursive_replace_layer(
        self,
        module: nn.Module,
        origin_cls: Union[str, nn.Module],
        attr_replacement: Dict[str, Any],
        param_replacement: List[Callable],
        method_replacement: Dict[str, Callable],
        sub_module_replacement: List[Callable],
    ) -> None:
        r"""
        Reverse the replace layer operation

        Args:
            layer (torch.nn.Module): The object of layer to shard
            origin_cls (Union[str, torch.nn.Module]): The origin layer class or a string of layer class name.
            attr_replacement (Dict): The attribute dict to modify
            param_replacement (List[Callable]): The function list to get parameter shard information in policy
            sub_module_replacement (List[Callable]): The function list to get sub module shard information in policy
        """
        if (isinstance(origin_cls, str) and origin_cls == module.__class__.__name__) or \
           (module.__class__ == origin_cls):
            if attr_replacement is not None:
                self._replace_attr(module, attr_replacement)

            if param_replacement is not None:
                self._replace_param(module, param_replacement)

            if method_replacement is not None:
                self._replace_method(module, method_replacement)

            if sub_module_replacement is not None:
                self._replace_sub_module(module, sub_module_replacement)

        for name, child in module.named_children():
            self._recursive_replace_layer(child, origin_cls, attr_replacement, param_replacement, method_replacement,
                                          sub_module_replacement)

    def _replace_attr(
        self,
        module: nn.Module,
        attr_replacement: Dict[str, Any],
    ) -> None:
        r"""
        Replace the attribute of the layer

        Args:
            layer (:class:`torch.nn.Module`): The object of layer to shard
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
            layer (:class:`torch.nn.Module`): The object of layer to shard
            param_replacement (List[Callable]): The function list to get parameter shard information in policy
        """
        for param_func in param_replacement:
            param_func(module)

    def _replace_method(self, module: nn.Module, method_replacement: Dict[str, Callable]):
        for method_name, new_method in method_replacement.items():
            # bind the new method to the module
            setattr(module, method_name, new_method.__get__(module, module.__class__))

    def _replace_sub_module(
        self,
        org_layer: nn.Module,
        sub_module_replacement: List[SubModuleReplacementDescription],
    ) -> None:
        r"""
        Shard one layer according to the policy, the layer should be the same class as the key in policy's argument_policy return dict

        Args:
            org_layer (torch.nn.Module): The origin layer object to shard
            sub_module_replacement (List[SubModuleReplacementDescription]): The sub module replacement description list

        """
        for description in sub_module_replacement:
            suffix = description.suffix
            target_module = description.target_module
            kwargs = {} if description.kwargs is None else description.kwargs

            assert target_module is not None, 'target_module should not be None'

            # TODO: support different parallel mode
            native_sub_module = getattr_(org_layer, suffix, ignore=True)

            assert not isinstance(native_sub_module, target_module), \
                f"The module with suffix {suffix} has been replaced, please check the policy"

            # if it is None and we are allowed to ignore this module
            # just skip
            if description.ignore_if_not_exist and native_sub_module is None:
                continue

            try:
                replace_layer = target_module.from_native_module(native_sub_module,
                                                                 self.shard_config.tensor_parallel_process_group,
                                                                 **kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to replace {suffix} of type {native_sub_module.__class__.__qualname__}"
                    f" with {target_module.__qualname__} with the exception: {e}. "
                    "Please check your model configuration or sharding policy, you can set up an issue for us to help you as well."
                )

            setattr_(org_layer, suffix, replace_layer)
