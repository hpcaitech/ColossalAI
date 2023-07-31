# part of code modified from https://github.com/tunib-ai/parallelformers

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type, Union

import torch.nn as nn

from ..shard.shard_config import ShardConfig

__all__ = ["ParallelModule", "SubModuleReplacementDescription", "ModulePolicyDescription", "Policy"]


class ParallelModule():

    def __init__(self):
        pass


@dataclass
class SubModuleReplacementDescription:
    r"""
    Describe how a submodule will be replaced

    Args:
        suffix (str): used to get the submodule object
        target_module (ParallelModule): specifies the module class used to replace to submodule
        kwargs (Dict[str, Any]): the dictionary used to pass extra arguments to the `ParallelModule.from_native_module` method.
        ignore_if_not_exist (bool): if the submodule does not exist, ignore it or raise an exception
    """
    suffix: str
    target_module: ParallelModule
    kwargs: Dict[str, Any] = None
    ignore_if_not_exist: bool = False


@dataclass
class ModulePolicyDescription:
    r"""
    Describe how the attributes and parameters will be transformed in a policy.

    Args:
        attribute_replacement (Dict[str, Any]): key is the attribute name, value is the attribute value after sharding
        param_replacement (List[Callable]): a list of functions to perform in-place param replacement. The function
                    must receive only one arguments: module. One example is

                    ```python
                    def example_replace_weight(module: torch.nn.Module):
                        weight = module.weight
                        new_weight = shard_rowwise(weight, process_group)
                        module.weight = torch.nn.Parameter(new_weight)
                    ```
        sub_module_replacement (List[SubModuleReplacementDescription]): each element in the list is a ParamReplacementDescription
                    object which specifies the module to be replaced and the target module used to replacement.
        method_replace (Dict[str, Callable]): key is the method name, value is the method for replacement
    """
    attribute_replacement: Dict[str, Any] = None
    param_replacement: List[Callable] = None
    sub_module_replacement: List[SubModuleReplacementDescription] = None
    method_replacement: Dict[str, Callable] = None


class Policy(ABC):
    r"""
    The base class for all the policies. For each different model, it should have a different policy class,
    like BertPolicy for Bert Model or OPTPolicy for OPT model.

    Shardformer has provided many built-in sharding policies for the mainstream models. You can use the
    built-in policies by setting `policy = None`, which is already the default argument for `Shardformer.optimize`.
    If you want to define your own policy, you can inherit from this class and overwrite the methods you want to modify.
    """

    def __init__(self) -> None:
        self.shard_config = None
        self.model = None
        self.shard_config = None

    def set_model(self, model: nn.Module) -> None:
        r"""
        Set model as an attribute of the Policy object so that we can access the model's attributes.

        Args:
            model (:class:`nn.Module`): The model to be perform
        """
        self.model = model

    def set_shard_config(self, shard_config: ShardConfig) -> None:
        r"""
        Set shard config as an attribute of the Policy object.

        Args:
            shard_config (:class:`ShardConfig`): The shard config to be perform
        """
        self.shard_config = shard_config
        self.config_sanity_check()

    @abstractmethod
    def config_sanity_check(self):
        """
        Check if the shard config is valid for the model. Raise an exception if the config is invalid.
        This method is made abstractmethod with no default implementation because we want to the policy writer
        to take note of the feature supported by his/her model and policy.
        """
        pass

    @abstractmethod
    def preprocess(self) -> nn.Module:
        r"""
        Perform some preprocessing of the model, like reshaping the embedding layer.
        """
        pass

    @abstractmethod
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        This method returns the module policy, which is a dictionary. The key is the module name or the module object,
        and the value is the ModulePolicyDescription object. The ModulePolicyDescription object describes how the module
        will be transformed.
        """
        pass

    @abstractmethod
    def postprocess(self) -> nn.Module:
        r"""
        Perform some postprocessing of the model, like binding the weight of embedding layer with
        the classifier layer
        """
        pass

    def append_or_create_submodule_replacement(
            self, description: Union[SubModuleReplacementDescription,
                                     List[SubModuleReplacementDescription]], policy: Dict[Union[str, nn.Module],
                                                                                          ModulePolicyDescription],
            target_key: Union[str, nn.Module]) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        Append or create a new submodule replacement description to the policy for the given key.

        Args:
            submodule_replace_desc (Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]]): the submodule replacement description to be appended
            policy (Dict[Union[str, nn.Module], ModulePolicyDescription]): the policy to be updated
            target_key (Union[str, nn.Module]): the key of the policy to be updated
        """
        # convert to list
        if isinstance(description, SubModuleReplacementDescription):
            description = [description]

        # append or create a new description
        if target_key in policy:
            policy[target_key].sub_module_replacement.extend(description)
        else:
            policy[target_key] = ModulePolicyDescription(sub_module_replacement=description)

        return policy
