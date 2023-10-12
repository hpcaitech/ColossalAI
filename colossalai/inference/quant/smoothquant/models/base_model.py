# Adapted from AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ

import os
import types
import warnings
from abc import abstractmethod
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers
from safetensors.torch import save_file as safe_save
from torch import device
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import PushToHubMixin, cached_file

from ....tensor_parallel.batch_infer_state import BatchInferState
from ....tensor_parallel.kvcache_manager import MemoryManager

CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = ["llama"]


def get_module_by_name_suffix(model, module_name: str):
    for name, module in model.named_modules():
        if name.endswith(module_name):
            return module


def simple_dispatch_model(model, device_map):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    if "" in device_map:
        d = device_map[""]
        model = model.to(torch.device(d))
        model.hf_device_map = device_map
        return model

    tied_params = accelerate.utils.modeling.find_tied_parameters(model)
    if set(device_map.values()) == {"cpu"} or set(device_map.values()) == {"cpu", "disk"}:
        main_device = "cpu"
    else:
        main_device = [d for d in device_map.values() if d not in ["cpu", "disk"]][0]

    cpu_offload_group = [(n, d) for n, d in device_map.items() if d == "cpu"]
    prev_hook = None
    for idx, (n, d) in enumerate(cpu_offload_group):
        m = get_module_by_name_suffix(model, n)
        _, prev_hook = accelerate.cpu_offload_with_hook(m, execution_device=main_device, prev_module_hook=prev_hook)
    # set first cpu offload module's prev_module_hook to the last cpu offload module's hook
    if len(cpu_offload_group) > 1:
        get_module_by_name_suffix(model, cpu_offload_group[0][0])._hf_hook.prev_module_hook = prev_hook

    for n, d in device_map.items():
        m = get_module_by_name_suffix(model, n)
        if d != "cpu":
            d = torch.device(d)
            hook = AlignDevicesHook(d, io_same_device=True, place_submodules=True)
            add_hook_to_module(m, hook)
    accelerate.utils.modeling.retie_parameters(model, tied_params)
    model.hf_device_map = device_map

    return model


class BaseSmoothForCausalLM(nn.Module, PushToHubMixin):
    layer_type: str = None

    def __init__(self, model: PreTrainedModel, quantized: bool = False):
        super().__init__()

        self.model = model
        self.model_type = self.model.config.model_type
        self._quantized = quantized
        self.config = self.model.config
        self.cache_manager = None
        self.max_total_token_num = 0

    @property
    def quantized(self):
        return self._quantized

    def init_cache_manager(self, max_total_token_num=2048):
        if self.config.model_type == "llama":
            head_num = self.config.num_key_value_heads
            layer_num = self.config.num_hidden_layers
            head_dim = self.config.hidden_size // head_num

        self.cache_manager = MemoryManager(max_total_token_num, torch.int8, head_num, head_dim, layer_num)
        self.max_total_token_num = max_total_token_num

    def init_batch_state(self, max_output_len=256, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size = len(input_ids)

        seq_start_indexes = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
        start_index = 0
        max_len_in_batch = -1

        for i in range(batch_size):
            seq_len = len(input_ids[i])
            seq_lengths[i] = seq_len
            seq_start_indexes[i] = start_index
            start_index += seq_len
            max_len_in_batch = seq_len if seq_len > max_len_in_batch else max_len_in_batch

        if "max_total_token_num" in kwargs.keys():
            max_total_token_num = kwargs["max_total_token_num"]
            self.init_cache_manager(max_total_token_num)

        if "max_new_tokens" in kwargs.keys():
            max_output_len = kwargs["max_new_tokens"]

        if batch_size * (max_len_in_batch + max_output_len) > self.max_total_token_num:
            max_total_token_num = batch_size * (max_len_in_batch + max_output_len)
            warnings.warn(f"reset max tokens to {max_total_token_num}")
            self.init_cache_manager(max_total_token_num)

        block_loc = torch.empty((batch_size, max_len_in_batch + max_output_len), dtype=torch.long, device="cuda")
        batch_infer_state = BatchInferState(batch_size, max_len_in_batch)
        batch_infer_state.seq_len = seq_lengths.to("cuda")
        batch_infer_state.start_loc = seq_start_indexes.to("cuda")
        batch_infer_state.block_loc = block_loc
        batch_infer_state.decode_layer_id = 0
        batch_infer_state.past_key_values_len = 0
        batch_infer_state.is_context_stage = True
        batch_infer_state.set_cache_manager(self.cache_manager)
        return batch_infer_state

    @abstractmethod
    @torch.inference_mode()
    def quantize(
        self,
        examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
    ):
        if self.quantized:
            raise EnvironmentError("can't execute quantize because the model is quantized.")

    def to(self, device: Union[str, torch.device]):
        self.model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, **kwargs):
        """shortcut for model.generate"""

        batch_infer_state = self.init_batch_state(**kwargs)
        if self.config.model_type == "llama":
            setattr(self.model.model, "infer_state", batch_infer_state)

        batch_infer_state.is_context_stage = True

        with torch.inference_mode():
            return self.model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def save_quantized(
        self,
        save_dir: str,
        model_basename: str,
        use_safetensors: bool = False,
        safetensors_metadata: Optional[Dict[str, str]] = None,
    ):
        """save quantized model and configs to local disk"""
        os.makedirs(save_dir, exist_ok=True)

        if not self.quantized:
            raise EnvironmentError("can only save quantized model, please execute .quantize first.")

        self.model.to(CPU)

        model_base_name = model_basename  # or f"smooth-"
        if use_safetensors:
            model_save_name = model_base_name + ".safetensors"
            state_dict = self.model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            if safetensors_metadata is None:
                safetensors_metadata = {}
            elif not isinstance(safetensors_metadata, dict):
                raise TypeError("safetensors_metadata must be a dictionary.")
            else:
                print(f"Received safetensors_metadata: {safetensors_metadata}")
                new_safetensors_metadata = {}
                converted_keys = False
                for key, value in safetensors_metadata.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        converted_keys = True
                        try:
                            new_key = str(key)
                            new_value = str(value)
                        except Exception as e:
                            raise TypeError(
                                f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                            )
                        if new_key in new_safetensors_metadata:
                            print(
                                f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                            )
                        new_safetensors_metadata[new_key] = new_value
                safetensors_metadata = new_safetensors_metadata
                if converted_keys:
                    print(
                        f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                    )

            # Format is required to enable Accelerate to load the metadata
            # otherwise it raises an OSError
            safetensors_metadata["format"] = "pt"

            safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
        else:
            model_save_name = model_base_name + ".bin"
            torch.save(self.model.state_dict(), join(save_dir, model_save_name))

        self.model.config.save_pretrained(save_dir)

    def save_pretrained(
        self,
        save_dir: str,
        use_safetensors: bool = False,
        safetensors_metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """alias of save_quantized"""
        warnings.warn("you are using save_pretrained, which will re-direct to save_quantized.")
        self.save_quantized(save_dir, use_safetensors, safetensors_metadata)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        **model_init_kwargs,
    ):
        if not torch.cuda.is_available():
            raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        # Parameters related to loading from Hugging Face Hub
        cache_dir = model_init_kwargs.pop("cache_dir", None)
        force_download = model_init_kwargs.pop("force_download", False)
        resume_download = model_init_kwargs.pop("resume_download", False)
        proxies = model_init_kwargs.pop("proxies", None)
        local_files_only = model_init_kwargs.pop("local_files_only", False)
        use_auth_token = model_init_kwargs.pop("use_auth_token", None)
        revision = model_init_kwargs.pop("revision", None)
        subfolder = model_init_kwargs.pop("subfolder", "")
        model_init_kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
        }

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **cached_file_kwargs)
        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype
        model_init_kwargs["trust_remote_code"] = trust_remote_code
        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.tie_weights()

            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
                low_zero=False,
            )
            model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
            )
            model_init_kwargs["low_cpu_mem_usage"] = True

            del model
        else:
            model_init_kwargs["device_map"] = None
            model_init_kwargs["low_cpu_mem_usage"] = False

        torch.cuda.empty_cache()

        merged_kwargs = {**model_init_kwargs, **cached_file_kwargs}
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **merged_kwargs)

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            warnings.warn("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return cls(model, False)

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        model_basename: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        use_safetensors: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """load quantized model from local disk"""

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        # == step1: prepare configs and file names == #
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code, **cached_file_kwargs
        )

        if config.model_type not in SUPPORTED_MODELS:
            raise TypeError(f"{config.model_type} isn't supported yet.")

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)
        is_local = isdir(model_name_or_path)

        resolved_archive_file = None
        if is_local:
            model_save_name = join(model_name_or_path, model_basename)
            for ext in extensions:
                if isfile(model_save_name + ext):
                    resolved_archive_file = model_save_name + ext
                    break
        else:  # remote
            for ext in extensions:
                resolved_archive_file = cached_file(model_name_or_path, model_basename + ext, **cached_file_kwargs)
                if resolved_archive_file is not None:
                    break

        if resolved_archive_file is None:  # Could not find a model file to use
            raise FileNotFoundError(f"Could not find model in {model_name_or_path}")

        model_save_name = resolved_archive_file

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights(include_buffers=True))

        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
            )
            if config.model_type == "llama":
                from .llama import LlamaSmoothquantDecoderLayer, init_to_get_rotary, llama_model_forward

                llama_config = model.config

                for i, layer in enumerate(model.model.layers):
                    model.model.layers[i] = LlamaSmoothquantDecoderLayer(llama_config)

                model.model.forward = types.MethodType(llama_model_forward, model.model)
                cos, sin = init_to_get_rotary(llama_config)
                model.model.register_buffer("_cos_cached", cos)
                model.model.register_buffer("_sin_cached", sin)
            model.tie_weights()

        accelerate.utils.modeling.load_checkpoint_in_model(
            model, checkpoint=model_save_name, offload_state_dict=True, offload_buffers=True
        )
        model = model.to("cuda")

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            warnings.warn("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        return cls(
            model,
            True,
        )

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            return getattr(self.model, item)


__all__ = ["BaseSmoothForCausalLM"]
