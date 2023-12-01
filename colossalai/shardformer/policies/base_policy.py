# part of code modified from https://github.com/tunib-ai/parallelformers

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from colossalai.pipeline.stage_manager import PipelineStageManager

from ..layer.normalization import BaseLayerNorm
from ..layer.parallel_module import ParallelModule
from ..shard.shard_config import ShardConfig

__all__ = ["ParallelModule", "SubModuleReplacementDescription", "ModulePolicyDescription", "Policy"]


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
    target_module: Union[ParallelModule, BaseLayerNorm]
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
        sub_module_replacement (List[SubModuleReplacementDescription]): each element in the list is a SubModuleReplacementDescription
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
        self.shard_config: Optional[ShardConfig] = None
        self.model: Optional[Module] = None

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

    @property
    def pipeline_stage_manager(self) -> Optional[PipelineStageManager]:
        if self.shard_config is not None:
            return self.shard_config.pipeline_stage_manager
        return None

    @abstractmethod
    def config_sanity_check(self):
        """
        Check if the shard config is valid for the model. Raise an exception if the config is invalid.
        This method is made abstractmethod with no default implementation because we want to the policy writer
        to take note of the feature supported by his/her model and policy.
        """

    @abstractmethod
    def preprocess(self) -> nn.Module:
        r"""
        Perform some preprocessing of the model, like reshaping the embedding layer.
        """

    @abstractmethod
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        This method returns the module policy, which is a dictionary. The key is the module name or the module object,
        and the value is the ModulePolicyDescription object. The ModulePolicyDescription object describes how the module
        will be transformed.
        """

    @abstractmethod
    def postprocess(self) -> nn.Module:
        r"""
        Perform some postprocessing of the model, like binding the weight of embedding layer with
        the classifier layer
        """

    def append_or_create_submodule_replacement(
        self,
        description: Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]],
        policy: Dict[Union[str, nn.Module], ModulePolicyDescription],
        target_key: Union[str, nn.Module],
    ) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
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
            if policy[target_key].sub_module_replacement is None:
                policy[target_key].sub_module_replacement = description
            else:
                policy[target_key].sub_module_replacement.extend(description)
        else:
            policy[target_key] = ModulePolicyDescription(sub_module_replacement=description)

        return policy

    def append_or_create_method_replacement(
        self,
        description: Dict[str, Callable],
        policy: Dict[Union[str, nn.Module], ModulePolicyDescription],
        target_key: Union[str, nn.Module],
    ) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        Append or create a new method replacement description to the policy for the given key.

        Args:
            description (Union[SubModuleReplacementDescription, List[SubModuleReplacementDescription]]): the submodule replacement description to be appended
            policy (Dict[Union[str, nn.Module], ModulePolicyDescription]): the policy to be updated
            target_key (Union[str, nn.Module]): the key of the policy to be updated
        """
        if target_key in policy:
            if policy[target_key].method_replacement is None:
                policy[target_key].method_replacement = description
            else:
                policy[target_key].method_replacement.update(description)
        else:
            policy[target_key] = ModulePolicyDescription(method_replacement=description)

        return policy

    def get_held_layers(self) -> List[Module]:
        """Get layers that should be held in current stage. This method should be implemented by subclass.

        Returns:
            List[Module]: List of layers that should be hold in current stage
        """
        raise NotImplementedError

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """Get parameters that should be shared across stages. This method should be implemented by subclass.

        Returns:
            List[Dict[int, Tensor]]: List of parameters that should be shared across stages. E.g. [{0: module.model.embed_tokens.weight, 3: module.lm_head.weight}]
        """
        return []

    @staticmethod
    def distribute_layers(num_layers: int, num_stages: int) -> List[int]:
        """Divide layers into stages"""
        quotient = num_layers // num_stages
        remainder = num_layers % num_stages

        # calculate the num_layers per stage
        layers_per_stage = [quotient] * num_stages

        # deal with the rest layers
        if remainder > 0:
            start_position = num_stages // 2 - remainder // 2
            for i in range(start_position, start_position + remainder):
                layers_per_stage[i] += 1
        return layers_per_stage

    @staticmethod
    def get_stage_index(
        layers_per_stage: List[int],
        stage: int,
        num_model_chunks: int = 1,
        num_stages: int = 0,
    ) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Get the start index and end index of layers for each stage.

        Args:
            layers_per_stage (List[int]): number of layers for each stage
            stage (int): the stage index
            num_stages (int): number of stages
            num_model_chunks (int): number of model chunks

        Returns:
            - Tuple[int, int]: the start index and end index of this stage
            - List[Tuple[int, int]]: the start index and end index of this stage for each model chunk

        """
        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        stage_indices = []
        for model_chunk in range(num_model_chunks):
            start_idx = num_layers_per_stage_accumulated[stage + model_chunk * num_stages]
            end_idx = num_layers_per_stage_accumulated[stage + model_chunk * num_stages + 1]
            stage_indices.append([start_idx, end_idx])

        return stage_indices[0] if num_model_chunks == 1 else stage_indices
