import copy
import os
from typing import Callable, Optional, Union

import torch
from torch.nn import Module

from colossalai.interface import pretrained as pretrained_interface


class PretrainedManager:
    old_from_pretrained: Optional[Callable] = None

    @staticmethod
    def inject() -> None:
        try:
            from transformers.modeling_utils import PreTrainedModel
        except ImportError:
            return
        # recover bound method to plain function
        PretrainedManager.old_from_pretrained = PreTrainedModel.from_pretrained.__func__
        PreTrainedModel.from_pretrained = new_from_pretrained

    @staticmethod
    def recover() -> None:
        try:
            from transformers.modeling_utils import PreTrainedModel
        except ImportError:
            return
        # convert plain function to class method
        PreTrainedModel.from_pretrained = classmethod(PretrainedManager.old_from_pretrained)
        PretrainedManager.old_from_pretrained = None


@classmethod
def new_from_pretrained(
    cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs
) -> Module:
    from transformers import GenerationConfig
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import ContextManagers, _add_variant, cached_file, has_file, is_offline_mode
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, logging

    from colossalai._compat import (
        auth_kwargs,
        get_download_url,
        get_no_init_weights,
        is_remote_url,
        is_safetensors_available,
    )

    # Transformers v5 compatibility helpers (see `colossalai/_compat.py`):
    # `no_init_weights` moved to `transformers.initialization`; `download_url` was removed
    # entirely and therefore returns `None`, which the raw-URL branch handles explicitly.
    no_init_weights = get_no_init_weights()
    download_url = get_download_url()

    logger = logging.get_logger(__name__)

    config = kwargs.pop("config", None)
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    # v4 called this argument `use_auth_token`; v5 renamed it to `token` (see
    # `colossalai/_compat.py`). Accept both names so an explicit v5 `token` does not
    # collide with the keyword passed to the config loader below.
    use_auth_token = kwargs.pop("use_auth_token", None)
    token = kwargs.pop("token", None)
    if use_auth_token is not None and token is not None and use_auth_token != token:
        raise ValueError("Conflicting token and use_auth_token arguments")
    if token is not None:
        use_auth_token = token
    revision = kwargs.pop("revision", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    kwargs.pop("_fast_init", True)
    torch_dtype = kwargs.pop("torch_dtype", None)
    dtype = kwargs.pop("dtype", None)
    if dtype is not None and torch_dtype is not None and dtype != torch_dtype:
        raise ValueError("Conflicting dtype and torch_dtype arguments")
    if dtype is not None:
        torch_dtype = dtype
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)

    kwargs.pop("state_dict", None)
    kwargs.pop("from_tf", False)
    kwargs.pop("from_flax", False)
    kwargs.pop("output_loading_info", False)
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("low_cpu_mem_usage", None)
    kwargs.pop("device_map", None)
    kwargs.pop("max_memory", None)
    kwargs.pop("offload_folder", None)
    kwargs.pop("offload_state_dict", False)
    kwargs.pop("load_in_8bit", False)
    kwargs.pop("load_in_4bit", False)
    kwargs.pop("quantization_config", None)
    kwargs.pop("adapter_kwargs", {})
    kwargs.pop("adapter_name", "default")
    kwargs.pop("use_flash_attention_2", False)

    use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

    if len(kwargs) > 0:
        logger.warning(f"Below kwargs may be ignored: {list(kwargs.keys())}")

    from_pt = True

    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            **auth_kwargs(use_auth_token),
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        config = copy.deepcopy(config)
        kwarg_attn_imp = kwargs.pop("attn_implementation", None)
        if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:
            config._attn_implementation = kwarg_attn_imp
        model_kwargs = kwargs

    if commit_hash is None:
        commit_hash = getattr(config, "_commit_hash", None)

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.

    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            if download_url is None:
                # Transformers v5 removed raw-URL loading: `download_url` and
                # `is_remote_url` no longer exist in the installed tree, and `cached_file`
                # no longer accepts URLs. Fail explicitly instead of treating the URL as a
                # repository id and exposing an opaque cache lookup error.
                raise NotImplementedError(
                    "Transformers v5 no longer supports loading weights from a raw URL; received "
                    f"{pretrained_model_name_or_path!r}. Download the weights to a local directory "
                    "first, or use a Hugging Face repository ID."
                )
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            if use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "local_files_only": local_files_only,
                    **auth_kwargs(use_auth_token),
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        pass
                    elif use_safetensors:
                        raise EnvironmentError(
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with `safetensors`. Please make sure that the model has been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                        )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        pass
                if resolved_archive_file is None:
                    # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                    # message.
                    has_file_kwargs = {
                        "revision": revision,
                        "proxies": proxies,
                        **auth_kwargs(use_auth_token),
                    }
                    if variant is not None and has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}"
                        )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                )

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
    else:
        resolved_archive_file = None

    if from_pt:
        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        dtype_orig = None

        if torch_dtype is not None:
            if not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"`torch_dtype` can be either `torch.dtype` or `None`, but received {torch_dtype}")
            dtype_orig = torch.get_default_dtype()
            torch.set_default_dtype(torch_dtype)

    try:
        config.name_or_path = pretrained_model_name_or_path
        init_contexts = [no_init_weights()]
        with ContextManagers(init_contexts):
            model = cls(config, *model_args, **model_kwargs)
    finally:
        if from_pt and dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    # If it is a model with generation capabilities, attempt to load the generation config
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **auth_kwargs(use_auth_token),
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        except (OSError, TypeError):
            logger.info("Generation config file not found, using a generation config created from the model config.")

    # set pretrained path
    if resolved_archive_file:
        pretrained_interface.set_pretrained_path(model, resolved_archive_file)

    return model
