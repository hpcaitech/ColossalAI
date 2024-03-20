import logging

import torch
import torch.nn as nn

from .bnb_config import BnbQuantizationConfig

try:
    import bitsandbytes as bnb
    IS_4BIT_BNB_AVAILABLE = bnb.__version__ >= "0.39.0"
    IS_8BIT_BNB_AVAILABLE = bnb.__version__ >= "0.37.2"
except ImportError:
    pass


logger = logging.getLogger(__name__)


def quantize_model(
    model: torch.nn.Module,
    bnb_quantization_config: BnbQuantizationConfig,
):
    """
    This function will quantize the input model with the associated config passed in `bnb_quantization_config`. If the
    model is in the meta device, we will load and dispatch the weights according to the `device_map` passed. If the
    model is already loaded, we will quantize the model and put the model on the GPU,

    Args:
        model (`torch.nn.Module`):
            Input model. The model can be already loaded or on the meta device
        bnb_quantization_config (`BnbQuantizationConfig`):
            The bitsandbytes quantization parameters
        weights_location (`str` or `os.PathLike`):
            The folder weights_location to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
            - a path to a folder containing a unique pytorch_model.bin file.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.

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

    modules_on_cpu = []
    # custom device map

    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    #if bnb_quantization_config.skip_modules is None:
    #    bnb_quantization_config.skip_modules = get_keys_to_not_convert(model)

    # add cpu modules to skip modules only for 4-bit modules
    if load_in_4bit:
        bnb_quantization_config.skip_modules.extend(modules_on_cpu)
    modules_to_not_convert = bnb_quantization_config.skip_modules

    # We add the modules we want to keep in full precision
    if bnb_quantization_config.keep_in_fp32_modules is None:
        bnb_quantization_config.keep_in_fp32_modules = []
    keep_in_fp32_modules = bnb_quantization_config.keep_in_fp32_modules
    modules_to_not_convert.extend(keep_in_fp32_modules)

    # compatibility with peft
    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit

    # assert model_device is cuda
    model_device = get_parameter_device(model)

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
                if module.bias is not None:
                    bnb_module.bias.data = module.bias.data
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

