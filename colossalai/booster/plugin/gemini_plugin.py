import gc
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import (
    get_model_base_filenames,
    get_optimizer_base_filenames,
    load_shard_state_dict,
    save_config_file,
    save_state_dict,
    save_state_dict_shards,
)
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.memory_tracer import MemStats

from .dp_plugin_base import DPPluginBase

__all__ = ["GeminiPlugin"]

SUPPORTED_PRECISION = ["fp16", "bf16"]
PRECISION_STR_TO_DTYPE = {"fp16": torch.half, "bf16": torch.bfloat16}


class GeminiCheckpointIO(GeneralCheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save sharded model to checkpoint but only on master process.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        As there is communication when getting state dict, model.state_dict() must be called on all processes.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_unsharded_model(self, model: GeminiDDP, checkpoint: str, strict: bool = True):
        """
        Load model from checkpoint with automatic unwrapping.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        super().load_unsharded_model(model, checkpoint, strict=strict)

    def save_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save unsharded optimizer state dict to checkpoint.
        After calling optimizer.state_dict(), the complete optimizer states will be collected on master rank.
        As there is communication when getting state dict, optimizer.state_dict() must be called on all processes.
        The saving process will only be executed by master rank.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"
        state_dict = optimizer.state_dict()
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def load_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str):
        """
        Loading unsharded optimizer from checkpoint file.
        For each process, only loading optimizer states of parameters it controls.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        super().load_unsharded_optimizer(optimizer, checkpoint)

    def save_sharded_model(
        self,
        model: GeminiDDP,
        checkpoint_path: str,
        gather_dtensor: bool = False,
        prefix: Optional[str] = None,
        max_shard_size: int = 1024,
        use_safetensors: bool = False,
    ):
        """
        Save sharded model.
        As there is communication when getting state dict, model.state_dict() must be called on all processes.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        if os.path.isfile(checkpoint_path):
            logging.error(f"Provided path ({checkpoint_path}) should be a directory, not a file")
            return

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        state_dict_shard = model.state_dict_shard(max_shard_size=max_shard_size, only_rank_0=True)
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint_path)

        # Save shards of optimizer states.
        is_master = self.coordinator.is_master()
        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint_path,
            index_file=index_file,
            base_filename=weights_name,
            is_master=is_master,
            use_safetensors=use_safetensors,
        )

        # only save the index file on the master rank
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            save_config_file(model.unwrap(), checkpoint_path)
            logging.info(
                f"The model is split into checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def load_sharded_model(
        self, model: GeminiDDP, checkpoint_index_file: Path, strict: bool = False, use_safetensors: bool = False
    ):
        """
        Load shard model, load model from multiple files.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        return super().load_sharded_model(model, checkpoint_index_file, strict, use_safetensors, load_sub_module=False)

    def save_sharded_optimizer(
        self, optimizer: GeminiOptimizer, checkpoint: Path, gather_dtensor: bool, prefix: str, size_per_shard: int
    ):
        """
        Save sharded optimizer state dict to checkpoint folder.
        As there is communication when getting state dict, this must be called on all processes.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Preparing file paths and index file.
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)

        # Store the information of param groups to param_group_file.
        index_file.append_meta_data("param_groups", param_group_file)
        group_file_path = os.path.join(checkpoint, param_group_file)
        param_groups = optimizer.get_param_groups_for_saving()
        torch.save(param_groups, group_file_path)

        # States are broken into shards within max_shard_size.
        state_dict_shard = optimizer.state_shard(prefix=prefix, max_shard_size=size_per_shard, only_rank_0=True)

        # Save shards of optimizer states.
        is_master = self.coordinator.is_master()
        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint,
            index_file=index_file,
            base_filename=states_name,
            is_master=is_master,
            use_safetensors=False,
        )

        # Wrap up index file. Only save it on master rank.
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            logging.info(
                f"The optimizer is going to be split to checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def load_sharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint_index_file: Path, prefix: str):
        """
        Loading sharded optimizer from checkpoint folder, with index file given.
        For each process, only loading optimizer states of parameters it controls.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        if not os.path.isfile(checkpoint_index_file):
            logging.error(f"Provided path ({checkpoint_index_file}) should be a file")

        assert isinstance(optimizer, GeminiOptimizer)

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)

        # Load param_groups.
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {checkpoint_index_file} for an optimizer. \
                               Lacking param group file under current directory."
            )
        saved_param_groups = torch.load(param_group_path)
        optimizer.load_param_groups(saved_param_groups)

        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        # Load optimizer states from shard files under checkpoint path.
        # For each file, only load the states managed by current process.
        for shard_file in checkpoint_files:
            state_dict_shard = load_shard_state_dict(Path(shard_file), use_safetensors=False)
            optimizer.load_param_states(state_dict_shard)
            del state_dict_shard
            gc.collect()

        optimizer.optimizer_loading_epilogue()

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)

    def save_lora_as_pretrained(
        self, model: Union[nn.Module, ModelWrapper], checkpoint: str, use_safetensors: bool = False
    ) -> None:
        """
        Save the lora adapters and adapter configuration file to checkpoint directory.
        This method is modified from PeftModel.save_pretrained of peft library to fit in Gemini.
        As there is communication when getting state dict, model.state_dict_for_lora() must be called on all processes.
        """
        from peft import PeftModel
        from peft.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
        from safetensors.torch import save_file as safe_save_file

        assert isinstance(model, ModelWrapper), "Please boost the model before saving!"
        peft_model = model.unwrap()
        assert isinstance(
            peft_model, PeftModel
        ), "The model doesn't have lora adapters, please enable lora before saving."

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        if self.coordinator.is_master():
            Path(checkpoint).mkdir(parents=True, exist_ok=True)
            peft_model.create_or_update_model_card(checkpoint)

        # Lora's adapter name is 'default'
        peft_config = peft_model.peft_config["default"]

        # Obtain state dict and save.
        output_state_dict = model.state_dict_for_lora(only_rank_0=True)
        if self.coordinator.is_master():
            if use_safetensors:
                safe_save_file(
                    output_state_dict,
                    os.path.join(checkpoint, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(output_state_dict, os.path.join(checkpoint, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = peft_model.base_model.model.__dict__.get("name_or_path", None)

        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = peft_model._get_base_model_class(
                is_prompt_tuning=peft_config.is_prompt_learning,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if self.coordinator.is_master():
            peft_config.save_pretrained(checkpoint, auto_mapping_dict=auto_mapping_dict)  # save the config
        peft_config.inference_mode = inference_mode


class GeminiPlugin(DPPluginBase):
    """
    Plugin for Gemini.

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import GeminiPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin = GeminiPlugin()

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)
    ```

    Args:
        chunk_config_dict (dict, optional): chunk configuration dictionary.
        chunk_init_device (torch.device, optional): device to initialize the chunk.
        placement_policy (str, optional): "static" and "auto". Defaults to "static".
        enable_gradient_accumulation (bool, optional): Whether to enable gradient accumulation. When set to True, gradient will be stored after doing backward pass. Defaults to False.
        shard_param_frac (float, optional): fraction of parameters to be sharded. Only for "static" placement.
            If `shard_param_frac` is 1.0, it's equal to zero-3. If `shard_param_frac` is 0.0, it's equal to zero-2. Defaults to 1.0.
        offload_optim_frac (float, optional): fraction of optimizer states to be offloaded. Only for "static" placement.
            If `shard_param_frac` is 1.0 and `offload_optim_frac` is 0.0, it's equal to old "cuda" placement. Defaults to 0.0.
        offload_param_frac (float, optional): fraction of parameters to be offloaded. Only for "static" placement.
            For efficiency, this argument is useful only when `shard_param_frac` is 1.0 and `offload_optim_frac` is 1.0.
            If `shard_param_frac` is 1.0, `offload_optim_frac` is 1.0 and `offload_param_frac` is 1.0, it's equal to old "cpu" placement.
            When using static placement, we recommend users to tune `shard_param_frac` first and then `offload_optim_frac`.
            Defaults to 0.0.
        warmup_non_model_data_ratio (float, optional): ratio of expected non-model data memory during warmup. Only for "auto" placement. Defaults to 0.8.
        steady_cuda_cap_ratio (float, optional): ratio of allowed cuda capacity for model data during steady state. Only for "auto" placement. Defaults to 0.9.
        precision (str, optional): precision. Support 'fp16' and 'bf16'. Defaults to 'fp16'.
        master_weights (bool, optional): Whether to keep fp32 master parameter weights in optimizer. Defaults to True.
        pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
        force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
        strict_ddp_mode (bool, optional): use strict ddp mode (only use dp without other parallelism). Defaults to False.
        search_range_m (int, optional): chunk size searching range divided by 2^20. Defaults to 32.
        hidden_dim (int, optional): the hidden dimension of DNN.
            Users can provide this argument to speed up searching.
            If users do not know this argument before training, it is ok. We will use a default value 1024.
        min_chunk_size_m (float, optional): the minimum chunk size divided by 2^20.
            If the aggregate size of parameters is still smaller than the minimum chunk size,
            all parameters will be compacted into one small chunk.
        memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".
            Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**16.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        max_norm (float, optional): max_norm used for `clip_grad_norm`. You should notice that you shall not do
            clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
        verbose (bool, optional): verbose mode. Debug info including chunk search result will be printed. Defaults to False.
    """

    def __init__(
        self,
        chunk_config_dict: Optional[dict] = None,
        chunk_init_device: Optional[torch.device] = None,
        placement_policy: str = "static",
        enable_gradient_accumulation: bool = False,
        shard_param_frac: float = 1.0,  # only for static placement
        offload_optim_frac: float = 0.0,  # only for static placement
        offload_param_frac: float = 0.0,  # only for static placement
        warmup_non_model_data_ratio: float = 0.8,  # only for auto placement
        steady_cuda_cap_ratio: float = 0.9,  # only for auto placement
        precision: str = "fp16",
        master_weights: bool = True,
        pin_memory: bool = False,
        force_outputs_fp32: bool = False,
        strict_ddp_mode: bool = False,
        search_range_m: int = 32,
        hidden_dim: Optional[int] = None,
        min_chunk_size_m: float = 32,
        memstats: Optional[MemStats] = None,
        gpu_margin_mem_ratio: float = 0.0,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert precision in SUPPORTED_PRECISION, f"precision {precision} is not supported"
        self.gemini_config = dict(
            chunk_config_dict=chunk_config_dict,
            chunk_init_device=(chunk_init_device or get_current_device()),
            placement_policy=placement_policy,
            enable_gradient_accumulation=enable_gradient_accumulation,
            shard_param_frac=shard_param_frac,
            offload_optim_frac=offload_optim_frac,
            offload_param_frac=offload_param_frac,
            warmup_non_model_data_ratio=warmup_non_model_data_ratio,
            steady_cuda_cap_ratio=steady_cuda_cap_ratio,
            pin_memory=pin_memory,
            force_outputs_fp32=force_outputs_fp32,
            strict_ddp_mode=strict_ddp_mode,
            search_range_m=search_range_m,
            hidden_dim=hidden_dim,
            min_chunk_size_m=min_chunk_size_m,
            memstats=memstats,
            mixed_precision=PRECISION_STR_TO_DTYPE[precision],
            master_weights=master_weights,
            enable_lora=False,
        )
        self.zero_optim_config = dict(
            gpu_margin_mem_ratio=gpu_margin_mem_ratio,
        )
        self.optim_kwargs = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        self.verbose = verbose
        self.lora_enabled = False

    def support_no_sync(self) -> bool:
        return False

    def support_lora(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return SUPPORTED_PRECISION

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ["cuda"]

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        if self.lora_enabled:
            from peft import PeftModel

            assert isinstance(
                model, PeftModel
            ), "The model should have been wrapped as a PeftModel when self.lora_enabled is True"

            self.gemini_config["enable_lora"] = True

            # The optimizer will be small when enabling lora, so no need to offload.
            self.gemini_config["offload_optim_frac"] = 0.0

        if not isinstance(model, ModelWrapper):
            # convert model to sync bn
            # FIXME(ver217): gemini does not support sync bn
            # In torch/nn/modules/_functions.py, line 22, ``mean, invstd = torch.batch_norm_stats(input, eps)`` will get fp32 mean and invstd even though the input is fp16.
            # This inconsistency of dtype will cause the error.
            # We have two possible solutions:
            # 1. keep batch norm always in fp32. This is hard for gemini, as it use chunks.
            # 2. patch sync bn or write a new on. This is relatively easy, but we need to test it.
            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model, None)

            # wrap the model with Gemini
            model = GeminiDDP(model, **self.gemini_config, verbose=self.verbose)

        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(
                optimizer, model, **self.zero_optim_config, **self.optim_kwargs, verbose=self.verbose
            )

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return GeminiCheckpointIO()

    def no_sync(self, model: nn.Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        raise NotImplementedError

    def enable_lora(
        self, model: nn.Module, pretrained_dir: Optional[str] = None, lora_config: Optional[Dict] = None
    ) -> nn.Module:
        from peft import PeftModel, get_peft_model

        assert not isinstance(model, GeminiDDP), "Lora should be enabled before boosting the model."
        self.lora_enabled = True

        if pretrained_dir is None:
            peft_model = get_peft_model(model, lora_config)
        else:
            peft_model = PeftModel.from_pretrained(model, pretrained_dir, is_trainable=True)

        # For parameters modules set to be fine-tuned and saved, their original copies don't participate in the calculation of loss.
        # Thus their requires_grad attribute should be manually set to False to avoid bugs(Peft set them to True after initialization).
        # e.g.: the 'classifier'/'score' modules in models for SequenceClassification
        modules_to_save = peft_model.modules_to_save
        if modules_to_save is not None:
            for n, p in peft_model.named_parameters():
                if any((f"{key}.original_module" in n) for key in modules_to_save):
                    p.requires_grad_(False)

        return peft_model
