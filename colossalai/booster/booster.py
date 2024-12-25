from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.logging import get_dist_logger

SUPPORT_PEFT = False
try:
    import peft

    SUPPORT_PEFT = True
except ImportError:
    pass

import colossalai.interface.pretrained as pretrained_utils
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.quantization import BnbQuantizationConfig

from .accelerator import Accelerator
from .mixed_precision import MixedPrecision, mixed_precision_factory
from .plugin import Plugin
from .plugin.pp_plugin_base import PipelinePluginBase

__all__ = ["Booster"]


class Booster:
    """
    Booster is a high-level API for training neural networks. It provides a unified interface for
    training with different precision, accelerator, and plugin.


    ```python
    # Following is pseudocode

    colossalai.launch(...)
    plugin = GeminiPlugin(...)
    booster = Booster(precision='fp16', plugin=plugin)

    model = GPT2()
    optimizer = HybridAdam(model.parameters())
    dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    lr_scheduler = LinearWarmupScheduler()
    criterion = GPTLMLoss()

    model, optimizer, criterion, dataloader, lr_scheduler = booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

    for epoch in range(max_epochs):
        for input_ids, attention_mask in dataloader:
            outputs = model(input_ids.cuda(), attention_mask.cuda())
            loss = criterion(outputs.logits, input_ids)
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    ```

    Args:
        device (str or torch.device): The device to run the training. Default: None.
                                      If plugin is not used or plugin doesn't control the device,
                                      this argument will be set as training device ('cuda' will be used if argument is None).
        mixed_precision (str or MixedPrecision): The mixed precision to run the training. Default: None.
                                If the argument is a string, it can be 'fp16', 'fp16_apex', 'bf16', or 'fp8'.
                                'fp16' would use PyTorch AMP while `fp16_apex` would use Nvidia Apex.
        plugin (Plugin): The plugin to run the training. Default: None.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        mixed_precision: Optional[Union[MixedPrecision, str]] = None,
        plugin: Optional[Plugin] = None,
    ) -> None:
        if plugin is not None:
            assert isinstance(
                plugin, Plugin
            ), f"Expected the argument plugin to be an instance of Plugin, but got {type(plugin)}."
        self.plugin = plugin
        self.logger = get_dist_logger()

        # set accelerator
        if self.plugin and self.plugin.control_device():
            self.accelerator = None
            if device is not None:
                self.logger.warning(
                    "The plugin will control the accelerator," "so the device argument will be ignored.", ranks=[0]
                )
        else:
            device = device or "cuda"
            self.accelerator = Accelerator(device)

        # set precision
        if self.plugin and self.plugin.control_precision():
            if mixed_precision is not None:
                self.logger.warning(
                    "The plugin will control the precision," "so the mixed_precision argument will be ignored.",
                    ranks=[0],
                )
            self.mixed_precision = None
        elif mixed_precision is None:
            self.mixed_precision = None
        else:
            # validate and set precision
            if isinstance(mixed_precision, str):
                # the user will take the default arguments for amp training
                self.mixed_precision = mixed_precision_factory(mixed_precision)
            elif isinstance(mixed_precision, MixedPrecision):
                # the user can customize the arguments by passing the precision object
                self.mixed_precision = mixed_precision
            else:
                raise ValueError(
                    f"Expected the argument mixed_precision to be a string or an instance of Precision, but got {type(mixed_precision)}."
                )

        if self.plugin is not None and self.plugin.control_checkpoint_io():
            self.checkpoint_io = self.plugin.get_checkpoint_io()
        else:
            self.checkpoint_io = GeneralCheckpointIO()

    def boost(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> List[Union[nn.Module, Optimizer, LRScheduler, DataLoader]]:
        """
        Wrap and inject features to the passed in model, optimizer, criterion, lr_scheduler, and dataloader.

        Args:
            model (nn.Module): Convert model into a wrapped model for distributive training.
                               The model might be decorated or partitioned by plugin's strategy after execution of this method.
            optimizer (Optimizer, optional): Convert optimizer into a wrapped optimizer for distributive training.
                                             The optimizer's param groups or states might be decorated or partitioned by plugin's strategy after execution of this method. Defaults to None.
            criterion (Callable, optional): The function that calculates loss. Defaults to None.
            dataloader (DataLoader, optional): The prepared dataloader for training. Defaults to None.
            lr_scheduler (LRScheduler, optional): The learning scheduler for training. Defaults to None.

        Returns:
            List[Union[nn.Module, Optimizer, LRScheduler, DataLoader]]: The list of boosted input arguments.
        """
        # TODO(FrankLeeeee): consider multi-model and multi-optimizer case
        # TODO(FrankLeeeee): consider multi-dataloader case
        pretrained_path = pretrained_utils.get_pretrained_path(model)
        # transform model for mixed precision
        if self.plugin:
            model, optimizer, criterion, dataloader, lr_scheduler = self.plugin.configure(
                model, optimizer, criterion, dataloader, lr_scheduler
            )

        if self.plugin and not self.plugin.control_device():
            # transform model for accelerator
            model = self.accelerator.configure_model(model)

        if self.mixed_precision and (self.plugin is None or self.plugin and not self.plugin.control_precision()):
            # transform model for mixed precision
            # when mixed_precision is specified and the plugin is not given or does not control the precision
            model, optimizer, criterion = self.mixed_precision.configure(model, optimizer, criterion)

        if pretrained_path:
            self.load_model(model, pretrained_path)
            # clear pretrained path attr
            orig_model = model.unwrap() if isinstance(model, ModelWrapper) else model
            pretrained_utils.set_pretrained_path(orig_model, None)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        """Execution of backward during training step.

        Args:
            loss (torch.Tensor): The loss for backpropagation.
            optimizer (Optimizer): The optimizer to be updated.
        """
        # TODO(frank lee): implement this method with plugin
        optimizer.backward(loss)

    def execute_pipeline(
        self,
        data_iter: Iterator,
        model: nn.Module,
        criterion: Callable[[Any, Any], torch.Tensor],
        optimizer: Optional[Optimizer] = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute forward & backward when utilizing pipeline parallel.
        Return loss or Huggingface style model outputs if needed.

        Warning: This function is tailored for the scenario of pipeline parallel.
        As a result, please don't do the forward/backward pass in the conventional way (model(input)/loss.backward())
        when doing pipeline parallel training with booster, which will cause unexpected errors.

        Args:
            data_iter(Iterator): The iterator for getting the next batch of data. Usually there are two ways to obtain this argument:
                                 1. wrap the dataloader to iterator through: iter(dataloader)
                                 2. get the next batch from dataloader, and wrap this batch to iterator: iter([batch])
            model (nn.Module): The model to execute forward/backward, it should be a model wrapped by a plugin that supports pipeline.
            criterion: (Callable[[Any, Any], torch.Tensor]): Criterion to be used. It should take two arguments: model outputs and inputs, and returns loss tensor.
                                                             'lambda y, x: loss_fn(y)' can turn a normal loss function into a valid two-argument criterion here.
            optimizer (Optimizer, optional): The optimizer for execution of backward. Can be None when only doing forward (i.e. evaluation). Defaults to None.
            return_loss (bool, optional): Whether to return loss in the dict returned by this method. Defaults to True.
            return_output (bool, optional): Whether to return Huggingface style model outputs in the dict returned by this method. Defaults to False.

        Returns:
            Dict[str, Any]: Output dict in the form of {'loss': ..., 'outputs': ...}.
                            ret_dict['loss'] is the loss of forward if return_loss is set to True, else None.
                            ret_dict['outputs'] is the Huggingface style model outputs during forward if return_output is set to True, else None.
        """
        assert isinstance(
            self.plugin, PipelinePluginBase
        ), f"The plugin {self.plugin.__class__.__name__} does not support pipeline."
        return self.plugin.execute_pipeline(data_iter, model, criterion, optimizer, return_loss, return_outputs)

    def no_sync(self, model: nn.Module = None, optimizer: OptimizerWrapper = None) -> contextmanager:
        """Context manager to disable gradient synchronization across DP process groups.
           Support torch DDP and Low Level ZeRO-1 for now.

        Args:
            model (nn.Module): The model to be disabled gradient synchronization, for DDP
            optimizer (OptimizerWrapper): The optimizer to be disabled gradient synchronization, for ZeRO1-1

        Returns:
            contextmanager: Context to disable gradient synchronization.
        """
        assert (
            self.plugin is not None
        ), f"no_sync is only enabled when a plugin is provided and the plugin supports no_sync."
        assert self.plugin.support_no_sync(), f"The plugin {self.plugin.__class__.__name__} does not support no_sync."
        return self.plugin.no_sync(model, optimizer)

    def enable_lora(
        self,
        model: nn.Module,
        pretrained_dir: Optional[str] = None,
        lora_config: "peft.LoraConfig" = None,
        bnb_quantization_config: Optional[BnbQuantizationConfig] = None,
        quantize=False,
    ) -> nn.Module:
        """
        Wrap the passed in model with LoRA modules for training. If pretrained directory is provided, lora configs and weights are loaded from that directory.
        Lora in ColossalAI is implemented using Huggingface peft library, so the arguments for Lora configuration are same as those of peft.

        Args:
            model (nn.Module): The model to be appended with LoRA modules.
            pretrained_dir(str, optional): The path to the pretrained directory, can be a local directory
                or model_id of a PEFT configuration hosted inside a model repo on the Hugging Face Hub.
                When set to None, create new lora configs and weights for the model using the passed in lora_config. Defaults to None.
            lora_config: (peft.LoraConfig, optional): Passed in LoraConfig for peft. Defaults to None.
        """
        if not SUPPORT_PEFT:
            raise ImportError("Please install Huggingface Peft library to enable lora features in ColossalAI!")

        assert self.plugin is not None, f"Lora can only be enabled when a plugin is provided."
        assert self.plugin.support_lora(), f"The plugin {self.plugin.__class__.__name__} does not support lora."
        if pretrained_dir is None:
            assert (
                lora_config is not None
            ), "Please provide configuration for Lora when pretrained directory path isn't passed in."
            assert isinstance(
                lora_config, peft.LoraConfig
            ), "The passed in configuration should be an instance of peft.LoraConfig."
        if lora_config is None:
            assert (
                pretrained_dir is not None
            ), "Please provide pretrained directory path if not passing in lora configuration."
        if quantize is True:
            if bnb_quantization_config is not None:
                self.logger.warning(
                    "User defined BnbQuantizationConfig is not fully tested in ColossalAI. Use it at your own risk.",
                    ranks=[0],
                )
            else:
                bnb_quantization_config = BnbQuantizationConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

        return self.plugin.enable_lora(model, pretrained_dir, lora_config, bnb_quantization_config)

    def load_model(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        strict: bool = True,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ) -> None:
        """Load model from checkpoint.

        Args:
            model (nn.Module or ModelWrapper): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Defaults to True.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """
        self.checkpoint_io.load_model(
            model, checkpoint, strict, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
        )

    def save_model(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
        use_async: bool = False,
    ) -> None:
        """Save model to checkpoint.

        Args:
            model (nn.Module or ModelWrapper): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It is a file path if ``shard=False``. Otherwise, it is a directory path.
            shard (bool, optional): Whether to save checkpoint a sharded way.
                If true, the checkpoint will be a folder with the same format as Huggingface transformers checkpoint. Otherwise, it will be a single file. Defaults to False.
            gather_dtensor (bool, optional): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            size_per_shard (int, optional): Maximum size of checkpoint shard file in MB. This is useful only when ``shard=True``. Defaults to 1024.
            use_safetensors (bool, optional): whether to use safe tensors. Default: False. If set to True, the checkpoint will be saved.
            use_async (bool, optional): whether to save the state_dict of model asynchronously. Default: False.
        """
        self.checkpoint_io.save_model(
            model,
            checkpoint=checkpoint,
            shard=shard,
            gather_dtensor=gather_dtensor,
            prefix=prefix,
            size_per_shard=size_per_shard,
            use_safetensors=use_safetensors,
            use_async=use_async,
        )

    def load_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ) -> None:
        """Load optimizer from checkpoint.

        Args:
            optimizer (Optimizer): An optimizer boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.
            low_cpu_mem_mode (bool): whether to load the model in low cpu memory mode. If false, it will use RAM cache to accelerate loading. Default: True.
            num_threads (int): number of threads to use when loading the model. Only useful when disabling low cpu mem mode. Default: 1.
        """
        self.checkpoint_io.load_optimizer(
            optimizer, checkpoint, low_cpu_mem_mode=low_cpu_mem_mode, num_threads=num_threads
        )

    def save_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_async: bool = False,
    ) -> None:
        """
        Save optimizer to checkpoint.

        Args:
            optimizer (Optimizer): An optimizer boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It is a file path if ``shard=False``. Otherwise, it is a directory path.
            shard (bool, optional): Whether to save checkpoint a sharded way.
                If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            size_per_shard (int, optional): Maximum size of checkpoint shard file in MB. This is useful only when ``shard=True``. Defaults to 1024.
        """
        self.checkpoint_io.save_optimizer(
            optimizer, checkpoint, shard, gather_dtensor, prefix, size_per_shard, use_async=use_async
        )

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str) -> None:
        """Save lr scheduler to checkpoint.

        Args:
            lr_scheduler (LRScheduler): A lr scheduler boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local file path.
        """
        self.checkpoint_io.save_lr_scheduler(lr_scheduler, checkpoint)

    def load_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str) -> None:
        """Load lr scheduler from checkpoint.

        Args:
            lr_scheduler (LRScheduler): A lr scheduler boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local file path.
        """
        self.checkpoint_io.load_lr_scheduler(lr_scheduler, checkpoint)

    def save_lora_as_pretrained(
        self, model: Union[nn.Module, ModelWrapper], checkpoint: str, use_safetensors: bool = False
    ) -> None:
        """
        Save the lora adapters and adapter configuration file to a pretrained checkpoint directory.

        Args:
            model (Union[nn.Module, ModelWrapper]): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint directory. It must be a local path.
            use_safetensors (bool, optional): Whether to use safe tensors when saving. Defaults to False.
        """
        if not SUPPORT_PEFT:
            raise ImportError("Please install Huggingface Peft library to enable lora features in ColossalAI!")
        assert self.plugin is not None, f"Lora can only be enabled when a plugin is provided."
        assert self.plugin.support_lora(), f"The plugin {self.plugin.__class__.__name__} does not support lora."
        self.checkpoint_io.save_lora_as_pretrained(model, checkpoint, use_safetensors)
