from typing import Any, Callable, Dict, List

import torch.nn as nn

from colossalai.cluster.process_group_manager import ProcessGroupManager

from ..policies.autopolicy import get_autopolicy
from ..policies.basepolicy import Policy, SubModuleReplacementDescription
from ..utils.utils import getattr_, setattr_
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

    def __init__(
            self,
            model: nn.Module,
            policy: Policy,
            shard_config: ShardConfig = None,    # TODO
            pg_manager: ProcessGroupManager = None) -> None:
        self.model = model
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.shard_config = shard_config
        self.pg_manager = pg_manager

    def shard(self) -> None:
        r"""
        Shard the model according to the policy
        """
        self.policy.set_model(self.model)
        self.policy.set_shard_config(self.shard_config)
        self._preprocess()
        self._replace_model_class()
        self._replace_module()
        self._postprocess()

    def reshape_embedding(self) -> None:
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        vocab_size = self.model_config.vocab_size
        world_size = self.shard_config.world_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
            self.model_config = self.model.config

    def _preprocess(self) -> None:
        self.model = self.policy.preprocess()

    def _postprocess(self) -> None:
        self.model = self.policy.postprocess()

    def _replace_model_class(self,) -> None:
        r"""
        Replace the model to policy defined model
        Mainly modify the forward and backward to fit distributed model

        e.g.
        ::
            BertForMaskedLM.forward -> BertForMaskedLM_.forward
        """
        new_model_class = self.policy.new_model_class()
        if new_model_class is None:
            return

        for key in new_model_class.__dict__.keys():
            if hasattr(self.model.__class__, key):
                setattr(
                    self.model.__class__,
                    key,
                    getattr(new_model_class, key),
                )

    def _replace_module(self,) -> None:
        r"""
        Replace the module according to the policy, and replace the module one by one

        Args:
            model (:class:`torch.nn.Module`): The model to shard
        """
        module_descriptions = self.policy.module_policy()
        for module_description in module_descriptions.items():
            origin_layer_cls = module_description[0]
            attr_replacement = module_description[1].attribute_replacement
            param_replacement = module_description[1].param_replacement
            sub_module_replacement = module_description[1].sub_module_replacement
            self._recursive_replace_layer(self.model, origin_layer_cls, attr_replacement, param_replacement,
                                          sub_module_replacement)

    def _recursive_replace_layer(
        self,
        module: nn.Module,
        origin_cls: nn.Module,
        attr_replacement: Dict[str, Any],
        param_replacement: List[Callable],
        sub_module_replacement: List[Callable],
    ) -> None:
        r"""
        Reverse the replace layer operation

        Args:
            layer (:class:`torch.nn.Module`): The object of layer to shard
            origin_cls (:class:`transformers.model`): The origin layer class
            attr_replacement (Dict): The attribute dict to modify
            param_replacement (List[Callable]): The function list to get parameter shard information in polic
            sub_module_replacement (List[Callable]): The function list to get sub module shard information in policy
        """
        if module.__class__ == origin_cls:
            self._replace_attr(module, attr_replacement)
            self._replace_param(module, param_replacement)
            self._replace_sub_module(module, sub_module_replacement)
        for name, child in module.named_children():
            self._recursive_replace_layer(child, origin_cls, attr_replacement, param_replacement,
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
        # TODO: support parameter shard
        pass

    def _replace_sub_module(
        self,
        org_layer: nn.Module,
        sub_module_replacement: List[SubModuleReplacementDescription],
    ) -> None:
        r"""
        Shard one layer according to the policy, the layer should be the same class as the key in policy's argument_policy return dict

        Args:
            org_layer (:class:`torch.nn.Module`): The origin layer object to shard
            param_funcs (:class:`List[typing.Callable]`): The function list to get shard information in policy class

        """
        for description in sub_module_replacement:
            suffix = description.suffix
            target_module = description.target_module
            kwargs = {} if description.kwargs is None else description.kwargs

            assert target_module is not None, 'target_module should not be None'

            # TODO: support different parallel mode
            native_sub_module = getattr_(org_layer, suffix)
            replace_layer = target_module.from_native_module(native_sub_module, self.pg_manager.pg_store['tp1d'],
                                                             **kwargs)

            setattr_(org_layer, suffix, replace_layer)
