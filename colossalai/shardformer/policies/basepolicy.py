# part of code modified from https://github.com/tunib-ai/parallelformers

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import torch.nn as nn


@dataclass
class Argument:
    r"""
    The argument class for the policy

    Args:
        attr_dict (Dict[str, Any]): The dict for the param setting
        param_funcs (:class:`List[Callable]`): The list for the param functions
    """
    attr_dict: Dict[str, Any]
    param_funcs: List[Callable]


@dataclass
class Layer:
    r"""
    The layer object for the policy

    Args:
        suffix: (str): the suffix of the layer.
        replace_layer (:class:`colosalai.nn`): The layer to replace the original layer
        ignore (bool): Whether to ignore this layer if it is not in the model
        reversed (bool): Whether the weight in layer is reversed, commonly the weight in `torch.nn.Linear` is [out, in],
                        but in GPT2 `Conv1D` layer is [in, out] which is reversed.
        n_cast (int): The number of weight will cast to, like q, k, v in attention layer, n_cast should be 3. commonly in TP, we just chunk the weight with the number of devices,
                        but in multi-head attention, we need to chunk the weight with the number of devices * n_head, and
                        each device should have a part of Q, K and V weight.
    """
    suffix: str = None
    replace_layer: Any = None
    ignore: bool = False
    reversed: bool = False
    n_cast: int = None


@dataclass
class Col_Layer(Layer):
    r"""
    Class for col shard layer in tensor parrallel

    Args:
        weight (str): The weight suffix of the layer
        bias (str): The bias suffix of the layer
        gather_output (bool): Whether to gather the output of the layer
    """
    weight: str = None
    bias: str = None
    gather_output: bool = False


@dataclass
class Row_Layer(Layer):
    r"""
    Class for col shard layer in tensor parrallel

    Args:
        weight (str): The weight suffix of the layer
        bias (str): The bias suffix of the layer
    """
    weight: str = None
    bias: str = None


@dataclass
class Dropout_Layer(Layer):
    r"""
    Class for dropout layer in tensor parrallel

    Args:
        p (str): The dropout rate suffix of the layer
    """
    p: str = None


class Policy():
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

    @staticmethod
    def argument_policy(model_config, world_size: int) -> Dict[nn.Module, Argument]:
        r"""
        Return the dict for the modify policy, the key is the original layer class and the value is the
        argument for the modify layer

        Args:
            model_config (:class:`tansformer.Config`): The config of transformer model
            world_size (int)): The world size of sharding model

        Return:
            Dict for the modify policy,
            ::
            {
                origin layer class1 (nn.Module): Argument(
                    attr_dict = {
                        argument1: value1,
                        argument2: value2,
                        ...
                    },
                    param_funcs = [
                        staticmethod1,
                        staticmethod2,
                        ...
                    ]
                ),
                origin layer class2 (nn.Module): Argument(
                    attr_dict = {
                        argument1: value1,
                        argument2: value2,
                        ...
                    },
                    param_funcs = [
                        staticmethod1,
                        staticmethod2,
                        ...
                    ]
                ),
                ...
            }

        """
        raise NotImplementedError

    @staticmethod
    def inject_policy() -> Union[Tuple[nn.Module, nn.Module], None]:
        r"""
        Return the dict for the inject model

        Return:
            The injected model, key is the original model and value is the new shardmodel
            ::
            (OrignModel, CustomModel)
            in `CustomModel`, we can overwrite the forward and backward process
        """
        return None

    @staticmethod
    def binding_policy() -> Union[Dict[str, str], None]:
        r"""
        Return the dict for the binding model, None means no need to bind

        Return:
            This method should return the binding relationship for some layers share the weight or bias,
            the key and value is the suffix of the weight or bias of the model
        ::
            return {
                "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
            }
        """
        return None

    @staticmethod
    def attn_in() -> Union[List, None]:
        r"""
        Attention qkv layer
        In this kind of method, we should return the list of ``Layer`` object, each ``Layer`` object should be
        ``Layer`` for no slicing, ``Col_Layer`` for col slicing, ``Row_Layer`` for row slicing. And the parameters
        in ``Layer`` object can refer to the ``Layer`` class.

        Returns:
            List[Layer]: List of layer object, each layer is the new
        """
        return None

    @staticmethod
    def attn_out() -> Union[List, None]:
        r"""
        Attention output projection layer

        Returns:
            List[Layer]: List of layer object
        """
        return None

    @staticmethod
    def mlp_in() -> Union[List, None]:
        r"""
        h -> 4h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return None

    @staticmethod
    def mlp_out() -> Union[List, None]:
        r"""
        4h -> h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return None

    @staticmethod
    def embedding() -> Union[List, None]:
        r"""
        Partially slice the embedding layer

        Return:
            List[Layer]: List of layer object
        """
        return None

    @staticmethod
    def unembedding() -> Union[List, None]:
        r"""
        Partially slice the embedding layer, None means there is no unembedding layer

        Return:
            List[Layer]: List of layer object
        """
        return None
