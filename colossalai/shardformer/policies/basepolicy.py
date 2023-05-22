# part of code modified from https://github.com/tunib-ai/parallelformers

import torch
import torch.nn as nn
import colossalai.nn as col_nn
from typing import Any, Dict, List, Type, Tuple, Callable
from transformers import AutoConfig
from dataclasses import dataclass, field

@dataclass
class Argument:
    attr_dict : Dict[str, Any]
    param_funcs : List[Callable]
    binding_layers : List[nn.Module] = field(default_factory=list)

@dataclass
class Layer:
    """
    The layer object for the policy

    Args:
        weight: The weight name of the layer
        bias: The bias name of the layer
        replace_layer: The layer to replace the original layer
        ignore: Whether to ignore this layer if it is not in the model
    """
    weight: str = None
    bias: str = None
    replace_layer: Any = None
    ignore: bool = False


@dataclass
class Col_Layer(Layer):
    """
    Class for col shard layer in MegatronLM
    """
    gather_output: bool = False


@dataclass
class Row_Layer(Layer):
    """
    Class for col shard layer in MegatronLM
    """
    pass


class Policy():
    """
    The base class for all the policies
    For each different model, it should have a different policy class, like BertPolicy for Bert Model 
    or OPTPolicy for OPT model. 
    AutoPolicy:
        shardformer already defined some policies for huggingface model, just set custom_policy = None
        to use the auto policy. In shardformer autopolicy, we define a base policy for one type model,
        like BertPolicy, and for each different Bert modle in huggingface like, BertForMaskedLM, 
        BertForSequenceClassification, etc., for each different Bert model we difine different policy class
        and overwrite the method inject_policy
    
    CustomPolicy:
    """
    @staticmethod
    def argument_policy(model_config, shard_config: int) -> Dict[nn.Module,Argument]:
        """
        Return a dict, the key is layer will be modified and the value is the Argument class with param setting and param functions

        Args:
            model_config: The config of transformer model
            shard_setting: The config of distributed model
        
        Return:
            Dict for the modify policy,
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
    def inject_policy() -> Tuple[nn.Module, nn.Module]:
        """
        Return the dict for the inject model 

        Return:
            The injected model, key is the original model and value is the new shardmodel
        """
        return ()
    

    @staticmethod
    def attn_in() -> List:
        """
        Attention qkv layer

        Returns:
            List[Layer]: List of layer object, each layer is the new 
        """
        return NotImplementedError


    @staticmethod
    def attn_out() -> List:
        """
        Attention output projection layer

        Returns:
            List[Layer]: List of layer object
        """
        return NotImplementedError


    @staticmethod
    def mlp_in() -> List:
        """
        h -> 4h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return NotImplementedError
        

    @staticmethod
    def mlp_out() -> List:
        """
        4h -> h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return NotImplementedError
        
    
    @staticmethod
    def embedding()->List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return NotImplementedError
        
    
    @staticmethod
    def unembedding()->List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return NotImplementedError
