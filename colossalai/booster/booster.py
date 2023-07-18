import warnings
from contextlib import contextmanager
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import GeneralCheckpointIO
from colossalai.interface import ModelWrapper

from .accelerator import Accelerator
from .mixed_precision import MixedPrecision, mixed_precision_factory
from .plugin import Plugin

__all__ = ['Booster']


class Booster:
    """
    Booster is a high-level API for training neural networks. It provides a unified interface for
    training with different precision, accelerator, and plugin.

    Examples:
        ```python
        colossalai.launch(...)
        plugin = GeminiPlugin(...)
        booster = Booster(precision='fp16', plugin=plugin)

        model = GPT2()
        optimizer = HybridAdam(model.parameters())
        dataloader = Dataloader(Dataset)
        lr_scheduler = LinearWarmupScheduler()
        criterion = GPTLMLoss()

        model, optimizer, lr_scheduler, dataloader = booster.boost(model, optimizer, lr_scheduler, dataloader)

        for epoch in range(max_epochs):
            for input_ids, attention_mask in dataloader:
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.logits, input_ids)
                booster.backward(loss, optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        ```

    Args:
        device (str or torch.device): The device to run the training. Default: 'cuda'.
        mixed_precision (str or MixedPrecision): The mixed precision to run the training. Default: None.
                                If the argument is a string, it can be 'fp16', 'fp16_apex', 'bf16', or 'fp8'.
                                'fp16' would use PyTorch AMP while `fp16_apex` would use Nvidia Apex.
        plugin (Plugin): The plugin to run the training. Default: None.
    """

    def __init__(self,
                 device: str = 'cuda',
                 mixed_precision: Union[MixedPrecision, str] = None,
                 plugin: Optional[Plugin] = None) -> None:
        if plugin is not None:
            assert isinstance(
                plugin, Plugin), f'Expected the argument plugin to be an instance of Plugin, but got {type(plugin)}.'
        self.plugin = plugin

        # set accelerator
        if self.plugin and self.plugin.control_device():
            self.accelerator = None
            warnings.warn('The plugin will control the accelerator, so the device argument will be ignored.')
        else:
            self.accelerator = Accelerator(device)

        # set precision
        if self.plugin and self.plugin.control_precision():
            warnings.warn('The plugin will control the precision, so the mixed_precision argument will be ignored.')
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
                    f'Expected the argument mixed_precision to be a string or an instance of Precision, but got {type(mixed_precision)}.'
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
        Boost the model, optimizer, criterion, lr_scheduler, and dataloader.

        Args:
            model (nn.Module): The model to be boosted.
            optimizer (Optimizer): The optimizer to be boosted.
            criterion (Callable): The criterion to be boosted.
            dataloader (DataLoader): The dataloader to be boosted.
            lr_scheduler (LRScheduler): The lr_scheduler to be boosted.
        """
        # TODO(FrankLeeeee): consider multi-model and multi-optimizer case
        # TODO(FrankLeeeee): consider multi-dataloader case
        # transform model for mixed precision
        if self.plugin:
            model, optimizer, criterion, dataloader, lr_scheduler = self.plugin.configure(
                model, optimizer, criterion, dataloader, lr_scheduler)

        if self.plugin and not self.plugin.control_device():
            # transform model for accelerator
            model = self.accelerator.configure(model)

        if self.mixed_precision and (self.plugin is None or self.plugin and not self.plugin.control_precision()):
            # transform model for mixed precision
            # when mixed_precision is specified and the plugin is not given or does not control the precision
            model, optimizer, criterion = self.mixed_precision.configure(model, optimizer, criterion)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        """Backward pass.

        Args:
            loss (torch.Tensor): The loss to be backpropagated.
            optimizer (Optimizer): The optimizer to be updated.
        """
        # TODO: implement this method with plugin
        optimizer.backward(loss)

    def execute_pipeline(self,
                         data_iter: Iterator,
                         model: nn.Module,
                         criterion: Callable[[torch.Tensor], torch.Tensor],
                         optimizer: Optimizer,
                         return_loss: bool = True,
                         return_outputs: bool = False) -> Tuple[Optional[torch.Tensor], ...]:
        # TODO: implement this method
        # run pipeline forward backward pass
        # return loss or outputs if needed
        pass

    def no_sync(self, model: nn.Module) -> contextmanager:
        """Context manager to disable gradient synchronization across DP process groups.

        Args:
            model (nn.Module): The model to be disabled gradient synchronization.

        Returns:
            contextmanager: Context to disable gradient synchronization.
        """
        assert self.plugin is not None, f'no_sync is only enabled when a plugin is provided and the plugin supports no_sync.'
        assert self.plugin.support_no_sync, f'The plugin {self.plugin.__class__.__name__} does not support no_sync.'
        return self.plugin.no_sync(model)

    def load_model(self, model: Union[nn.Module, ModelWrapper], checkpoint: str, strict: bool = True):
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

    def save_model(self,
                   model: Union[nn.Module, ModelWrapper],
                   checkpoint: str,
                   shard: bool = False,
                   gather_dtensor: bool = True,
                   prefix: Optional[str] = None,
                   size_per_shard: int = 1024,
                   use_safetensors: bool = False):
        """Save model to checkpoint.

        Args:
            model (nn.Module or ModelWrapper): A model boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local path.
                It is a file path if ``shard=False``. Otherwise, it is a directory path.
            shard (bool, optional): Whether to save checkpoint a sharded way.
                If true, the checkpoint will be a folder. Otherwise, it will be a single file. Defaults to False.
            gather_dtensor (bool, optional): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Defaults to None.
            size_per_shard (int, optional): Maximum size of checkpoint shard file in MB. This is useful only when ``shard=True``. Defaults to 1024.
            use_safetensors (bool, optional): whether to use safe tensors. Default: False. If set to True, the checkpoint will be saved.
        """
        self.checkpoint_io.save_model(model,
                                      checkpoint=checkpoint,
                                      shard=shard,
                                      gather_dtensor=gather_dtensor,
                                      prefix=prefix,
                                      size_per_shard=size_per_shard,
                                      use_safetensors=use_safetensors)

    def load_optimizer(self, optimizer: Optimizer, checkpoint: str):
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

    def save_optimizer(self,
                       optimizer: Optimizer,
                       checkpoint: str,
                       shard: bool = False,
                       gather_dtensor: bool = True,
                       prefix: Optional[str] = None,
                       size_per_shard: int = 1024):
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

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """Save lr scheduler to checkpoint.

        Args:
            lr_scheduler (LRScheduler): A lr scheduler boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local file path.
        """
        self.checkpoint_io.save_lr_scheduler(lr_scheduler, checkpoint)

    def load_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """Load lr scheduler from checkpoint.

        Args:
            lr_scheduler (LRScheduler): A lr scheduler boosted by Booster.
            checkpoint (str): Path to the checkpoint. It must be a local file path.
        """
        self.checkpoint_io.load_lr_scheduler(lr_scheduler, checkpoint)
