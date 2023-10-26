import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

SUPPORT_PEFT = False
try:
    import peft

    SUPPORT_PEFT = True
except ImportError:
    pass

import colossalai.interface.pretrained as pretrained_utils
from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper

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

        # set accelerator
        if self.plugin and self.plugin.control_device():
            self.accelerator = None
            if device is not None:
                warnings.warn("The plugin will control the accelerator, so the device argument will be ignored.")
        else:
            device = device or "cuda"
            self.accelerator = Accelerator(device)

        # set precision
        if self.plugin and self.plugin.control_precision():
            if mixed_precision is not None:
                warnings.warn("The plugin will control the precision, so the mixed_precision argument will be ignored.")
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
        task_type: Optional[Union["peft.TaskType", str]] = None,
        inference_mode: bool = False,
        r: int = 8,
        target_modules: Optional[Union[List[str], str]] = None,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        bias: str = "none",
        modules_to_save: Optional[List[str]] = None,
        layers_to_transform: Optional[Union[List[int], int]] = None,
        layers_pattern: Optional[str] = None,
    ) -> nn.Module:
        """
        Wrap the passed in model with LoRA modules for training.
        Lora in ColossalAI is implemented using Huggingface peft library, so the arguments for Lora configuration are identical to those of peft.

        Args:
            model (nn.Module): The model to be appended with LoRA modules.
            task_type (Union[peft.TaskType, str], optional): The type of task to perform in peft. Available task types in string include "SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM",
                "TOKEN_CLS", "QUESTION_ANS", and "FEATURE_EXTRACTION". Defaults to None.
            inference_mode (bool, optional): Whether to use the Peft model in inference mode. Defaults to False.
            r (int, optional): Lora attention dimension. Defaults to 8.
            target_modules (Union[List[str],str], optional): List of names or regex expressions of the modules to apply Lora to.
                For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. Defaults to None.
            lora_alpha (int, optional): The alpha parameter for Lora scaling. Defaults to 8.
            lora_dropout (float, optional): The dropout probability for Lora layers. Defaults to 0.0.
            fan_in_fan_out (bool, optional): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
                For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set
                to `True`. Defaults to False.
            bias (str, optional): Bias type for Lora. Can be 'none', 'all' or 'lora_only'. If 'all' or 'lora_only', the
                corresponding biases will be updated during training. Be aware that this means that, even when disabling
                the adapters, the model will not produce the same output as the base model would have without adaptation.
                Defaults to "none".
            modules_to_save (List[str], optional):List of modules apart from LoRA layers to be set as trainable
                and saved in the final checkpoint. Defaults to None.
            layers_to_transform (Union[List[int],int], optional): The layer indexes to transform, if this argument is specified,
                it will apply the LoRA transformations on the layer indexes that are specified in this list. If a single integer
                is passed, it will apply the LoRA transformations on the layer at this index. Defaults to None.
            layers_pattern (str, optional): The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
                pattern is not in the common layers pattern. Defaults to None.
        """
        if not SUPPORT_PEFT:
            raise ImportError("Please install Huggingface Peft library to enable lora features in ColossalAI!")
        assert self.plugin is not None, f"Lora can only enabled when a plugin is provided."
        assert self.plugin.support_lora(), f"The plugin {self.plugin.__class__.__name__} does not support lora."
        lora_config = dict(
            task_type=task_type,
            inference_mode=inference_mode,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            bias=bias,
            modules_to_save=modules_to_save,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )
        return self.plugin.enable_lora(model, lora_config)

    def load_model(self, model: Union[nn.Module, ModelWrapper], checkpoint: str, strict: bool = True) -> None:
        """Load model from checkpoint.

        Args:
            model (nn.Module or ModelWrapper): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Defaults to True.
        """
        self.checkpoint_io.load_model(model, checkpoint, strict)

    def save_model(
        self,
        model: Union[nn.Module, ModelWrapper],
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
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
        """
        self.checkpoint_io.save_model(
            model,
            checkpoint=checkpoint,
            shard=shard,
            gather_dtensor=gather_dtensor,
            prefix=prefix,
            size_per_shard=size_per_shard,
            use_safetensors=use_safetensors,
        )

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str) -> None:
        """Load optimizer from checkpoint.

        Args:
            optimizer (Optimizer): An optimizer boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It should be a directory path if the checkpoint is sharded. Otherwise, it should be a file path.
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            size_per_shard (int, optional): Maximum size of checkpoint shard file in MB. This is useful only when ``shard=True``. Defaults to 1024.
        """
        self.checkpoint_io.load_optimizer(optimizer, checkpoint)

    def save_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
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
        self.checkpoint_io.save_optimizer(optimizer, checkpoint, shard, gather_dtensor, prefix, size_per_shard)

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

    def save_lora(self, model: Union[nn.Module, ModelWrapper], checkpoint: str, use_safetensors: bool = False) -> None:
        """
        Save the lora adapters and adapter configuration file to checkpoint directory.

        Args:
            model (Union[nn.Module, ModelWrapper]): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint directory. It must be a local path.
            use_safetensors (bool, optional): Whether to use safe tensors when saving. Defaults to False.
        """
        if not SUPPORT_PEFT:
            raise ImportError("Please install Huggingface Peft library to enable lora features in ColossalAI!")
        self.checkpoint_io.save_lora(model, checkpoint, use_safetensors)

    def load_lora(self, model: Union[nn.Module, ModelWrapper], checkpoint: str) -> None:
        """
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Args:
            model (Union[nn.Module, ModelWrapper]): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint directory. It must be a local path.
            use_safetensors (bool, optional): Whether to use safe tensors when saving. Defaults to False.
        """
        if not SUPPORT_PEFT:
            raise ImportError("Please install Huggingface Peft library to enable lora features in ColossalAI!")
        self.checkpoint_io.load_lora(model, checkpoint)
