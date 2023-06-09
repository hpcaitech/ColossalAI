# ⚡️ ShardFormer

## 📚 Table of Contents

- [⚡️ ShardFormer](#️-shardformer)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [🔨 Usage](#-usage)
  - [🔮 Simple example](#-simple-example)
  - [💡 Policy](#-policy)
  - [😊 Module](#-module)


## 🔗 Introduction

**Shardformer** is a module that automatically parallelizes the mainstream models in libraries such as HuggingFace and TIMM. This module aims to make parallelization hassle-free for users who are not from the system background.

## 🔨 Usage

The sample API usage is given below:

``` python
from colossalai.shardformer import shard_model
from transformers import BertForMaskedLM

# create huggingface model as normal
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# make the huggingface model paralleled to ShardModel
# auto policy:
sharded_model = shard_model(model)

# custom policy:
from xxx import <POLICYCLASS>
sharded_model = shard_model(model, <POLICYCLASS>)

# do angthing as normal
...
```

## 🔮 Simple example

``` shell
# inference
colossalai run --nproc_per_node 2 --master_port 29500 test.py --config config.py --mode inference
# train
colossalai run --nproc_per_node 2 --master_port 29500 test.py --config config.py --mode train
```


## 💡 Policy

If you wanna parallel the model in a custom way, just overwrite the policy class for the Hugging Face model.

You should do:

1. Inherit Policy class
2. Overwrite `argument_policy` method
    - In this method, you need to list which layers class you wanna modify and the attributes and parameters in those layers. Shardformer will replace all the layer belonging to the class you specified.
    - `attr_dict` is dict contains all the attributes need to be modified in this layer.
    - `param_funcs` is a list contains some functions which will return the path of the weight and bias from the layer.
3. Overwrite `inject_policy` method (Optional)
    - Shardformer will inject the model according to this method. If you need to modify the forward or backward progress (like distributed corssentropy loss in Bert) you need to overwrite this method.
4. Overwrite or add the param functions
    - These functions use a suffix to record the path of weight or bias for the layer.
    - The return is a list contains some `Col_Layer` or `Row_Layer` objects, which means slice along col and row respectively.
5. Overwrite `binding_policy` (Optional)
    - Overwrite to specify Shardformer will bind some weight between layers, like embedding and unembedding layers.
    - This function will return a dict, the key and value are the suffix of weight need to be binded.

More details can be found in shardformer/policies/basepolicy.py
``` python
from colossalai.shardformer.policies.basepolicy import Policy, Layer, Col_Layer, Row_Layer, Argument

CustomPolicy(Policy):
@staticmethod
    def argument_policy(model_config, shard_config: int) -> Dict[nn.Module, Argument]:
        r"""
        Return the dict for the modify policy, the key is the original layer class and the value is the
        argument for the modify layer

        Args:
            model_config (:class:`tansformer.Config`): The config of transformer model
            shard_config (:class:`ShardConfig`): The config for sharding model

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
    def inject_policy() -> Tuple[nn.Module, nn.Module]:
        r"""
        Return the dict for the inject model

        Return:
            The injected model, key is the original model and value is the new shardmodel
            ::
            (OrignModel, CustomModel)
            in `CustomModel`, we can overwrite the forward and backward process
        """
        return ()

    @staticmethod
    def binding_policy() -> Dict:
        r"""
        Return the dict for the binding model

        Return:
            This method should return the binding relationship for some layers share the weight or bias,
            the key and value is the suffix of the weight or bias of the model
        ::
            return {
                "bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight",
            }
        """
        return NotImplementedError

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
    def embedding() -> List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return NotImplementedError

    @staticmethod
    def unembedding() -> List:
        """
        Partially slice the embedding layer
        vocab_size->vocab_size//gpu_nums

        Return:
            List[Layer]: List of layer object
        """
        return NotImplementedError

```


## 😊 Module

  1. Flowchart

  <p align="center">
      <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/shardformer/shardformer_flowchart.png" width="600" />
  </p>

  2. Important Modules

  - CLASS `shard_model`:

    This is the user api to use shardformer, just create a model from transformers and define a custom policy or use shardformer autopolicy to make a shard model.

  - CLASS `Layer`:

    Parameters:
    - weight (str): The weight suffix of the layer
    - bias (str): The bias suffix of the layer
    - replace_layer (:class:`colosalai.nn`): The layer to replace the original layer
    - ignore (bool): Whether to ignore this layer if it is not in the model

    This class is used to specify the replacement policy for a particular layer. If `replace_layer` is None, only parameter partitioning will be performed without replacing the layer class.

    CLASS `Col_Layer(Layer)`:
      - gather_output (bool): Whether to gather the output of the layer

      This class inherited from `Layer`, representing the layer will be sliced along column.

    CLASS `Row_Layer(Layer)`:

      This class inherited from `Layer`, representing the layer will be sliced along row.

  - CLASS `Policy`:

    In Shardformer, this class holds significant importance as it defines the model partitioning methods, required parameter modifications, and model injection techniques all within a single Policy class.
    - `Policy.attn_in()/attn_out()/mlp_in()/mlp_out()/embedding()/unembedding()`......

      These functions define the partitioning methods of the parameters at different locations in the model. Each function returns a list of objects of Layer class that specify the replacement approach for these parameters. Shardformer also supports user-defined functions for modifying their models, in addition to the listed functions.
    - `Policy.argument_policy()`

      In this function, the user should use multiple dict to define which class of layers will require replacement. This includes the attributes and parameters that need to be modified or replaced. Attributes are stored in the form of a "suffix-string: value" dict, while parameters are stored via multiple static methods that return the replacement approach.
    - `Policy.inject_policy()`

      This function will return the injected model to replace the original model. The new model should be a nn.Module class which includes modified forward or backward functions or anything else.
    - `Policy.binding_policy()`

      This function will return the weight sharing information in the model in some dict. The key and value are both the suffixes of the shared parameters.

  - CLASS `ModelSharder(model, policy)`:

    This class helps shard the model, the parameter is the created transformers model and the custom policy. If custom policy is None, shardformer will automatically get already defined policy for the model.
    - `ModelShard.inject_model()`

      This function is used to inject the model to modify the forward and backward progress.
    - `ModelShard.replace_layer()`

      This function is used to replace the original layers with colossalai layer to make them paralleled and can do distributed communication.
    - `ModelShard.bind_layer()`

      This function is used to help different layers share weight or bias.

  - CLASS `Slicer`:

    This class is used to slice tensor according to policy.


  3. DistCrossEntropy Loss
  - Overview

    In order to reduce the communication size, caculate the crossentropy before all gather, refer to [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), reduce the communication size from [batch_size * seq_length * vocab_size] to [batch_size * seq_length]. The origin loss function is:
    $$ loss = -\log(\frac{\exp(x[class])}{\sum_i\exp(x[i])})$$

    alse can be represented as:

    $$ loss = \log(\sum_i\exp(x[i])) - x[class]$$

  - Step

    - First get the maximum logits across all the devices, make all the logist minus the maximun value to scale the value less than zero to avoid the value of exp being too large

    - Get a mask to mask the logits not in the local device

    - Caculate the loss according to the second formula
