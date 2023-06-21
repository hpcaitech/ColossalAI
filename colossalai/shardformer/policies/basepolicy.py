# part of code modified from https://github.com/tunib-ai/parallelformers

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type, Union

import torch.nn as nn

from ..shard.shard_config import ShardConfig


class ParallelModule():

    def __init__(self):
        pass


@dataclass
class SubModuleReplacementDescription:
    r"""
    Describe how a submodule will be replaced

    suffix (str): used to get the submodule object
    target_module (ParallelModule): specifies the module class used to replace to submodule
    kwargs (Dict[str, Any]): the dictionary used to pass extra arguments to the `ParallelModule.from_native_module` method.
    """
    suffix: str
    target_module: ParallelModule
    kwargs: Dict[str, Any] = None
    ignore_if_not_exist: bool = False


@dataclass
class ModulePolicyDescription:
    r"""
    Describe how the attributes and parameters will be transformed in a policy

    attribute_replacement (Dict[str, Any]): key is the attribute name, value is the attribute value after sharding
    param_replacement (List[Callable]): a list of functions to perform in-place param replacement. The function
    must receive two arguments: module, process_group. One example is

    ```python
    def example_replace_weight(module: torch.nn.Module, process_group):
        weight = module.weight
        new_weight = shard_rowwise(weight, process_group)
        module.weight = torch.nn.Parameter(new_weight)
    ```

    sub_module_replacement: each element in the list is a ParamReplacementDescription object which specifies
    the module to be replaced and the target module used to replacement
    """
    attribute_replacement: Dict[str, Any]
    param_replacement: List[Callable]
    sub_module_replacement: List[SubModuleReplacementDescription]


class Policy(ABC):
    r"""
    The base class for all the policies

    For each different model, it should have a different policy class, like BertPolicy for Bert Model
    or OPTPolicy for OPT model.

    AutoPolicy:
        Shardformer already defined some policies for huggingface model, just set ``custom_policy`` = None
        to use the auto policy. In shardformer autopolicy, we define a base policy for one type model,
        like BertPolicy, and for each different Bert modle in huggingface like, BertForMaskedLM,
        BertForSequenceClassification, etc., for each different Bert model we difine different policy class
        and overwrite the method like ``inject_policy`` to modify the forward and backward process.

    CustomPolicy:
        If you want to define your own policy, you can set ``custom_policy`` = CustomPolicy, and overwrite
        all the methods in ``Policy`` class. You can refer to any policy we defined like the ``BertPolicy``
        class for the example.

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

    @abstractmethod
    def preprocess(self) -> nn.Module:
        r"""
        Perform some preprocessing of the model, like reshaping the embedding layer
        """
        pass

    @abstractmethod
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        r"""
        Return the dict for the modify policy, the key is the original layer class and the value is the
        argument for the modify layer

        Return:
            Dict for the modify policy,
            ::
            {
                origin layer class1 (nn.Module): ModulePolicyDescription(
                    attribute_replacement = {
                        "attribute1": value1,
                        "attribute2": value2,
                        ...
                    },
                    param_replacement = [
                        function1,
                        function2,
                        ...
                    ],
                    sub_module_replacement = [
                        `SubModuleReplacementDescription` description1,
                        `SubModuleReplacementDescription` description2,
                        ...
                    ]
                ),
                origin layer class2 (nn.Module): ModulePolicyDescription(
                    ...
                ),
                ...
            }
        """
        pass

    @abstractmethod
    def new_model_class(self) -> Union[Type[nn.Module], None]:
        r"""
        Return the new model class for the new model, None means no need to modify the model class

        Return:
            New model class

            E.g.
            ```
            return BertModel_
            ```
        """
        pass

    @abstractmethod
    def postprocess(self) -> nn.Module:
        r"""
        Perform some postprocessing of the model, like binding the weight of embedding layer with
        the classifier layer
        """
        pass
