# ⚡️ ShardFormer

## 📚 Table of Contents

- [⚡️ ShardFormer](#️-shardformer)
  - [📚 Table of Contents](#-table-of-contents)
  - [🔗 Introduction](#-introduction)
  - [🔨 Usage](#-usage)
    - [Quick Start](#quick-start)
    - [Write your own policy](#write-your-own-policy)
  - [🗺 Roadmap](#-roadmap)
  - [💡 API Design](#-api-design)
    - [Distributed Modules](#distributed-modules)
    - [Shard Config](#shard-config)
    - [Policy](#policy)
    - [Model Sharder](#model-sharder)
    - [User-facing API](#user-facing-api)
  - [⌨️ Development Notes](#️-development-notes)
    - [Add New Policy to Shardformer](#add-new-policy-to-shardformer)
    - [Write Your Unit Testing](#write-your-unit-testing)
  - [📊 Benchmarking](#-benchmarking)
    - [System Performance](#system-performance)
    - [Convergence](#convergence)


## 🔗 Introduction

**Shardformer** is a module that automatically parallelizes the mainstream models in libraries such as HuggingFace and TIMM. This module aims to make parallelization hassle-free for users who are not from the system background.

## 🔨 Usage

### Quick Start

The sample API usage is given below:

``` python
from colossalai.shardformer import ShardConfig, Shard
from transformers import BertForMaskedLM

# launch colossalai
colossalai.launch_from_torch()

# create model
config = BertConfig.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

# create huggingface model as normal
shard_config = ShardConfig()
shard_former = ShardFormer(shard_config=shard_config)
sharded_model = shard_former.optimize(model).to('cuda')

# do everything like normal
...
```

### Write your own policy

If you have a custom model, you can also use Shardformer to parallelize it by writing your own sharding policy. More information about the sharding policy can be found in [API Design](#-api-design).

```python
from colossalai.shardformer import Policy

class MyPolicy(Policy):
    # implement your own policy
    ...

# init model and shard former
...

# use customized policy to shard model
my_policy = MyPolicy()
shard_former.optimize(model, my_policy)



```
## 🗺 Roadmap

We will follow this roadmap to develop Shardformer:

- [x] API Design
- [x] API Implementation
- [x] Unit Testing
- [ ] Policy Implementation
  - [ ] Hugging Face
    - [ ] NLP
      - [x] BERT
      - [x] T5
      - [x] LlaMa
      - [x] GPT2
      - [x] OPT
      - [x] BLOOM
      - [ ] GLM
      - [ ] RoBERTa
      - [ ] ALBERT
      - [ ] ERNIE
      - [ ] GPT Neo
      - [ ] GPT-J
    - [ ] CV
      - [x] ViT
      - [ ] BEiT
      - [ ] SwinTransformer
      - [ ] SwinTransformer V2
    - [ ] Audio
      - [ ] Whisper
    - [ ] Multi-modal
      - [ ] To be added

## 💡 API Design

We will discuss the major components of `ShardFormer` below to help you better understand how things work.
This section serves as the design doc for Shardformer and the function signature might differ from the actual implementation.
Please refer to the code for more details.

<p align="center">
   <img src="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/shardformer/shardformer_flowchart.png" width="600" />
   <br/>
</p>



### Distributed Modules

`ShardFormer` replaces the original PyTorch module with a distributed module.
The distributed module keeps the same attributes as the original module but replaces the original parameters with distributed parameters and defines a new `forward` function to execute distributed computation.
Each distributed module implements its `from_native_module` static method to convert the PyTorch module to its corresponding distributed module.

```python
class ParallelModule(torch.nn.Module):

    @abstractmethod
    def from_native_module(module: torch.nn.Module, process_group: Union[ProcessGroup, Tuple[ProcessGroup]]) -> ParallelModule
        """
        Convert a native module to a parallelized

        Examples:

        ```python
        # replace module
        my_linear = Linear1D_Col.from_native_module(my_linear, process_group)
        ```
        """
```

### Shard Config

`ShardConfig` is a simple data class to tell `ShardFormer` how sharding will be performed.

```python
@dataclass
class ShardConfig:
    tensor_parallel_process_group: ProcessGroup = None
    enable_fused_normalization: bool = False
    ...

    # Some possible future config fields
    tensor_parallel_mode: Choice['1d', '2d', '2.5d', '3d'] # support different tensor parallel mode
    inference_only: bool # only inject inference-suitable sharding policy
    use_flash_attention: bool # whether to use flash attention to speed up attention
```

### Policy

The `Policy` class describes how to handle the model sharding.
It is merely a description, the actual sharding will be performed by `ModelSharder`.
We abstract the policy into four stages:

1. Preprocessing: call `Policy.preprocess` to do some prior work before sharding, for example, resizing the embedding
2. Providing `ModulePolicyDescription`: call `Policy.module_policy` to get a bunch of `ModulePolicyDescription` to tell `ModelSharder` how the submodules's attributes, child parameters, and deeper submodules will be substituted.
3. Postprocessing: call `Policy.postprocess` to perform some postprocessing work, for example, binding the embedding and classifier head weights of the BERT model.

``` python
@dataclass
class ModulePolicyDescription:
    r"""
    Describe how the attributes and parameters will be transformed in a policy.

    Args:
        attribute_replacement (Dict[str, Any]): key is the attribute name, value is the attribute value after sharding
        param_replacement (List[Callable]): a list of functions to perform in-place param replacement. The function must receive only one arguments: module.
        sub_module_replacement (List[SubModuleReplacementDescription]): each element in the list is a ParamReplacementDescription
                    object which specifies the module to be replaced and the target module used to replacement.
        method_replace (Dict[str, Callable]): key is the method name, value is the method for replacement
    """
    attribute_replacement: Dict[str, Any] = None
    param_replacement: List[Callable] = None
    sub_module_replacement: List[SubModuleReplacementDescription] = None
    method_replacement: Dict[str, Callable] = None

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


class Policy(ABC):

    def __init__(self)
        self.model = None

    def set_model(self, model: nn.Module) -> None:
        """
        Set model as an attribute of the Policy object so that we can access the model's attributes.
        """
        self.model = model

    @abstractmethod
    def preprocess(self) -> nn.Module:
        """
        Perform some preprocessing on the model, such as resizing the embedding size
        """
        ...

    @abstractmethod
    def module_policy(self) -> Dict[Union[str, nn.Module], ModulePolicyDescription]:
        """
        Return the dict for the modify policy, the key is the original layer class and the value is the
        argument for the modify layer
        """
        ...

    @abstractmethods
    def postprocess(self) -> nn.Module:
        """
        Perform some postprocessing on the model, such as binding the embedding with the weight of the classifier head
        """
        ...
```


### Model Sharder

`ModelSharder` is the class in charge of sharding the model based on the given policy.

```python
class ModelSharder:

    def __init__(self, model: torch.nn.Module, shard_config: ShardConfig, Policy: ShardPolicy = None):
        #TODO: input is a cls or a obj
        ...

    def shard(self) -> None:
        """
        Shard model with parallelism with the help of pre-processing, replace_model_class, replace_module, and post-processing.
        """
        ...

    def replace_module(self) -> None:
        """
        Replace the layer according to the policy. Call Policy.module_policy() to get the module. Call _replace_module recursively.
        """
        ...
```

### User-facing API

We only expose a limited number of APIs to the user to keep their user experience simple and clean.

```python
class ShardFormer:
    """
    Parallelize model based on the given config and policy

    Example:

    shard_former = ShardFormer(shard_config=shard_config)
    shard_former.init_distributed()
    model = shard_former.optimize(model, policy=policy)
    dataloader = shard_former.shard_dataset(dataset)

    """

    def __init__(self, shard_config: ShardConfig):
        """
        Do two things:
        1. Create a colossalai.cluster.process_group_manager to manage process groups for dp, tp and pp
        2. serve as a store for shard config
        """
        self.shard_config = shard_config
        self.pg_manager = None

    def init_distributed(self) -> colossalai.cluster.ProcessGroupManager:
        """
        Initialize the distributed process group according to the
        """
        pg_manager = ...
        self.pg_manager = pg_manager
        return pg_manager

    def shard_model(self, model: torch.nn.Module，policy: Policy) -> torch.nn.Module:
        """
        Shard model for TP and PP
        """
        ...

    def shard_dataset(self, dataset: Dataset) -> Dataloader:
        """
        Shard dataset for DP
        """
        ...
```

## ⌨️ Development Notes

### Add New Policy to Shardformer

This section serves as the guideline for writing new policies and register them into `shardformer`.

- Step 1. Write your own model policy

You can create a new file in the `colossalai/shardformer/policies` folder and name the file with the model name. You can implement your policy in this file. You should not import the any model zoo library at the header section of the file because we do not want to import the library when we do not use the policy. Libraries such as `transformers` should be imported only in the function body when needed.

Please follow the following protocols when writing your policy:

- You have to make a clear decision what you want to replace exactly in the original PyTorch module
   - Use `ModulePolicyDescription.attribute_replacement` to replace the module attributes
   - Use `ModulePolicyDescription.param_replacement` to replace the module parameters
   - Use `ModulePolicyDescription.sub_module_replacement` to replace the submodules completely. The target module should implement the `from_native_module` for the .
   - Use `ModulePolicyDescription.method_replacement` to replace the module methods. **These replacement methods should be put in the `shardformer/modeling/<model-name>.py`**.
- You can implement the `ParallelModule` for primitive modules in the `shardformer/layer/<model-name>.py` file. Primitive modules refer to modules which are not composed of other modules. For example, the `torch.nn.Linear` module is a primitive module while modules such as `BertEncoder` module in the `transformers` library is a composite module. Primitive modules do not nested inner `nn.Module` members. For composite modules, you should consider using `ModulePolicyDescription` to implement your replacement.
- `ParallelModule` is meant to be used in two ways: `ParallelModule.from_native_module` to convert native PyTorch module to the `ParallelModule` and `ParallelModule(...)` to instantiate the module directly just like a normal PyTorch module. `ParallelModule` should be only implemented for modules whose weights are sharded. If you want to make your module compatible with the `ModulePolicyDescription.sub_module_replacement` and there is no weight sharding in your module, you can just implement the `from_native_module` method without inheriting the `ParallelModule` like `colossalai/shardformer/layer/normalization.py`.
- **Do not import any file in the `colossalai/shardformer/policies` and `colossalai/shardformer/modeling` to avoid unwanted import error**. For example, a file in these folders accidentally imports `transformers` library at the top of the file, then the user will have to install `transformers` library even if they do not use this file. Any file in the `modeling` folder should be only imported by the policy file. A policy implementation should be only imported dynamically via the autopolicy or manually via the `ShardFormer` module.
- Try to keep your import statement on third-party libraries such as `transformers` within the function body instead of the header section of the file. This is because we do not want to import the library when we do not use the policy.


- Step 2. Register your policy to the autopolicy

Next, you need to register your policy in the `colossalai/shardformer/policies/autopolicy.py` file.

For example, if we register the policy for the BERT model, we just add a key-value in the `_POLICY_LIST` dictionary. The key if the `qualname` of the model object (you can get it by model.__class__.__qualname__). The value is a `PolicyLocation` object, which contains the file name and the class name of the policy. We do not import the policy directly because the policy file may contain libraries (such as `transformers`) which we do not want to import when we do not use the policy.

```python
_POLICY_LIST = {
    # BERT
    "transformers.models.bert.modeling_bert.BertModel":
        PolicyLocation(file_name="bert", class_name="BertModelPolicy"),
}
```

### Write Your Unit Testing

This section serves as the guideline for testing the `shardformer` module.

- Step 1. Add your model to the model zoo in the test kits.

Add your model to the `tests/kit/model_zoo` file. This allows you to define test-related components for this model. You can take `tests/kit/model_zoo/transformers/llama.py` as an example for reference.

- Step 2. Write your unit testing for the model

Next, implement your unit test in the `tests/test_shardformer` folder. Please refer to other similar tests for style consistency.


- Step 3. Execute your test

When you run tests locally, you should run tests for both your newly-added test file and the whole `shardformer` module tests.

```bash
# test for your own test file
pytest tests/test_shardformer/test_model/<your-file>.py

# test for the whole shardformer module
pytest tests/test_shardformer
```

## 📊 Benchmarking

### System Performance

To be added.

### Convergence

To validate that training the model using shardformers does not impact its convergence. We [fine-tuned the BERT model](./examples/shardformer_benchmark.py) using both shardformer and non-shardformer approaches. We compared the accuracy, loss, F1 score of the training results.

| accuracy |   f1    |  loss   | GPU number | model shard |
| :------: | :-----: | :-----: | :--------: | :---------: |
| 0.82594  | 0.87441 | 0.09913 |     4      |    True     |
| 0.81884  | 0.87299 | 0.10120 |     2      |    True     |
| 0.81855  | 0.87124 | 0.10357 |     1      |    False    |

Overall, the results demonstrate that using shardformers during model training does not affect the convergence.
