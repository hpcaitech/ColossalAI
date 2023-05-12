# part of code modified from https://github.com/tunib-ai/parallelformers

import torch
import torch.nn as nn
import colossalai.nn as col_nn
from typing import Any, Dict, List, Type, Tuple
from transformers import AutoConfig
from dataclasses import dataclass

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
    weight: str
    bias: str
    replace_layer: Any
    ignore: bool = False

class Policy():
    """
    The base class for all the policies
    """
    def __init__(
            self,
            replace_layer: nn.Module
            ) -> None:
        """
        Init the policy class
        
        Args:
            inject_layer: Layer the policy will apply to
        """
        self.replace_layer = replace_layer

    @staticmethod
    def argument_policy(config, dist_setting: int) -> Dict[nn.Module, Dict]:
        """
        Return the argument and its value need to be modified

        Args:
            config: The config of transformer model
            dist_setting: The setting of distributed model
        
        Return:
            Dict for the modify policy,
            {
                origin_layer1 (nn.Module): {argument1: value1, argument2: value2 ...},
                origin_layer2 (nn.Module): {argument1: value1, argument2: value2 ...},
                ...
            }

        """
        return {}
    

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
        return []

    @staticmethod
    def attn_out() -> List:
        """
        Attention output projection layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

    @staticmethod
    def mlp_in() -> List:
        """
        h -> 4h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return []

    @staticmethod
    def mlp_out() -> List:
        """
        4h -> h mlp layer

        Returns:
            List[Layer]: List of layer object
        """
        return []
    
    @staticmethod
    def embedding()->List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return []
    
    @staticmethod
    def unembedding()->List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return []


    # @staticmethod
    # def original_layer_class() -> Type[nn.Module]:
    #     """
    #     Class to apply the policy to
    #     e.g. BertLayer, GPT2Block, BartEncoderLayer, ...

    #     Returns:
    #         Type[nn.Module]: original layer class
    #     """
    #     raise NotImplementedError

