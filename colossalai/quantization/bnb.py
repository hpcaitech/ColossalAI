# adapted from Hugging Face accelerate/utils/bnb.py accelerate/utils/modeling.py

import importlib.metadata
import logging

import torch
import torch.nn as nn
from packaging.version import Version

from .bnb_config import BnbQuantizationConfig

try:
    import bitsandbytes as bnb

    try:
        # in case lower version of bitsandbytes does not have __version__ attribute
        BNB_VERSION = Version(bnb.__version__)
    except AttributeError:
        BNB_VERSION = Version(importlib.metadata.version("bitsandbytes"))

    IS_4BIT_BNB_AVAILABLE = BNB_VERSION >= Version("0.39.0")
    IS_8BIT_BNB_AVAILABLE = BNB_VERSION >= Version("0.37.2")
except ImportError:
    pass


logger = logging.getLogger(__name__)


def quantize_model(
    model: torch.nn.Module,
    bnb_quantization_config: BnbQuantizationConfig,
):
    """
    This function will quantize the input loaded model with the associated config passed in `bnb_quantization_config`.
    We will quantize the model and put the model on the GPU.

    Args:
        model (`torch.nn.Module`):
            Input model. The model already loaded
        bnb_quantization_config (`BnbQuantizationConfig`):
            The bitsandbytes quantization parameters

    Returns:
        `torch.nn.Module`: The quantized model
    """

    load_in_4bit = bnb_quantization_config.load_in_4bit
    load_in_8bit = bnb_quantization_config.load_in_8bit

    if load_in_8bit and not IS_8BIT_BNB_AVAILABLE:
        raise ImportError(
            "You have a version of `bitsandbytes` that is not compatible with 8bit quantization,"
            " make sure you have the latest version of `bitsandbytes` installed."
        )
    if load_in_4bit and not IS_4BIT_BNB_AVAILABLE:
        raise ValueError(
            "You have a version of `bitsandbytes` that is not compatible with 4bit quantization,"
            "make sure you have the latest version of `bitsandbytes` installed."
        )

    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    if bnb_quantization_config.skip_modules is None:
        bnb_quantization_config.skip_modules = get_keys_to_not_convert(model)

    modules_to_not_convert = bnb_quantization_config.skip_modules

    # We add the modules we want to keep in full precision
    if bnb_quantization_config.keep_in_fp32_modules is None:
        bnb_quantization_config.keep_in_fp32_modules = []
    keep_in_fp32_modules = bnb_quantization_config.keep_in_fp32_modules

    # compatibility with peft
    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit

    # assert model_device is cuda
    model_device = next(model.parameters()).device

    model = replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert)

    # convert param to the right dtype
    dtype = bnb_quantization_config.torch_dtype
    for name, param in model.state_dict().items():
        if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
            param.to(torch.float32)
            if param.dtype != torch.float32:
                name = name.replace(".weight", "").replace(".bias", "")
                param = getattr(model, name, None)
                if param is not None:
                    param.to(torch.float32)
        elif torch.is_floating_point(param):
            param.to(dtype)
    if model_device.type == "cuda":
        # move everything to cpu in the first place because we can't do quantization if the weights are already on cuda
        model.cuda(torch.cuda.current_device())
        torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        model.to(torch.cuda.current_device())
        logger.info(
            f"The model device type is {model_device.type}. However, cuda is needed for quantization."
            "We move the model to cuda."
        )
    else:
        raise RuntimeError("No GPU found. A GPU is needed for quantization.")
    return model


def replace_with_bnb_layers(model, bnb_quantization_config, modules_to_not_convert=None, current_key_name=None):
    """
    A helper function to replace all `torch.nn.Linear` modules by `bnb.nn.Linear8bit` modules or by `bnb.nn.Linear4bit`
    modules from the `bitsandbytes`library. The function will be run recursively and replace `torch.nn.Linear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[str]`):
            Names of the modules to not quantize convert. In practice we keep the `lm_head` in full precision for
            numerical stability reasons.
        current_key_name (`List[str]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert.
    """

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    model, has_been_replaced = _replace_with_bnb_layers(
        model, bnb_quantization_config, modules_to_not_convert, current_key_name
    )
    if not has_been_replaced:
        logger.warning(
            "You are loading your model in 8bit or 4bit but no linear modules were found in your model."
            " this can happen for some architectures such as gpt2 that uses Conv1D instead of Linear layers."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model


def _replace_with_bnb_layers(
    model,
    bnb_quantization_config,
    modules_to_not_convert=None,
    current_key_name=None,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    # bitsandbytes will initialize CUDA on import, so it needs to be imported lazily

    has_been_replaced = False
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            proceed = True
            for key in modules_to_not_convert:
                if (
                    (key in current_key_name_str) and (key + "." in current_key_name_str)
                ) or key == current_key_name_str:
                    proceed = False
                    break
            if proceed:
                # Load bnb module with empty weight and replace ``nn.Linear` module
                if bnb_quantization_config.load_in_8bit:
                    bnb_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=bnb_quantization_config.llm_int8_threshold,
                    )
                elif bnb_quantization_config.load_in_4bit:
                    bnb_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        bnb_quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=bnb_quantization_config.bnb_4bit_use_double_quant,
                        quant_type=bnb_quantization_config.bnb_4bit_quant_type,
                    )
                else:
                    raise ValueError("load_in_8bit and load_in_4bit can't be both False")
                bnb_module.weight.data = module.weight.data
                bnb_module.weight.skip_zero_check = True
                if module.bias is not None:
                    bnb_module.bias.data = module.bias.data
                    bnb_module.bias.skip_zero_check = True
                bnb_module.requires_grad_(False)
                setattr(model, name, bnb_module)
                has_been_replaced = True
        if len(list(module.children())) > 0:
            _, _has_been_replaced = _replace_with_bnb_layers(
                module, bnb_quantization_config, modules_to_not_convert, current_key_name
            )
            has_been_replaced = has_been_replaced | _has_been_replaced
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def get_keys_to_not_convert(model):
    r"""
    An utility function to get the key of the module to keep in full precision if any For example for CausalLM modules
    we may want to keep the lm_head in full precision for numerical stability reasons. For other architectures, we want
    to keep the tied weights of the model. The function will return a list of the keys of the modules to not convert in
    int8.

    Parameters:
    model (`torch.nn.Module`):
        Input model
    """
    # Create a copy of the model
    # with init_empty_weights():
    #    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model = model

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = sum(list(tied_params.values()), []) + list(tied_params.keys())
    else:
        tied_keys = sum(tied_params, [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = False
    if hasattr(model, "base_model_prefix"):
        is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_children())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


def find_tied_parameters(model: nn.Module, **kwargs):
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """
    # Initialize result and named_parameters before recursing.
    named_parameters = kwargs.get("named_parameters", None)
    prefix = kwargs.get("prefix", "")
    result = kwargs.get("result", {})

    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        # A tied parameter will not be in the full `named_parameters` seen above but will be in the `named_parameters`
        # of the submodule it belongs to. So while recursing we track the names that are not in the initial
        # `named_parameters`.
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return FindTiedParametersResult([sorted([weight] + list(set(tied))) for weight, tied in result.items()])


class FindTiedParametersResult(list):
    """
    This is a subclass of a list to handle backward compatibility for Transformers. Do not rely on the fact this is not
    a list or on the `values` method as in the future this will be removed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def values(self):
        return sum([x[1:] for x in self], [])
