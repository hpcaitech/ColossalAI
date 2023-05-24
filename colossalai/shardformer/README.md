# ⚡️ ShardFormer

## 📚 Table of Contents

- [⚡️ ShardFormer](#️-shardformer)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [🔨 Usage](#-usage)
  - [🔮 Simple example](#-simple-example)
  - [💡 Policy](#-policy)

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
2. Overwrite argument_policy method
    - In this method, you need to list which layers class you wanna modify and the attributes and parameters in those layers.
3. Overwrite inject_policy method (Optional)
    - If you need to modify the forward or backward progress.
4. Overwrite or add the param recording functions
    - These functions use a suffix to record the path of weight or bias for the layer.
5. Overwrite binding

More details can be found in shardformer/policies/basepolicy.py
``` python
from colossalai.shardformer.policies.basepolicy import Policy, Layer, Col_Layer, Row_Layer, Argument

CustomPolicy(Policy):
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
    def binding_policy() -> Dict:
        """
        Return the dict for the binding model
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
