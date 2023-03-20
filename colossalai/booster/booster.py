from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from .accelerator import Accelerator
from .mixed_precision import MixedPrecision, mixed_precision_factory
from .plugin import Plugin

__all__ = ['Booster']


class Booster:
    """
    Booster is a high-level API for training neural networks. It provides a unified interface for
    training with different precisio, accelerator, and plugin.

    Examples:
        >>> colossalai.launch(...)
        >>> plugin = GeminiPlugin(stage=3, ...)
        >>> booster = Booster(precision='fp16', plugin=plugin)
        >>>
        >>> model = GPT2()
        >>> optimizer = Adam(model.parameters())
        >>> dataloader = Dataloader(Dataset)
        >>> lr_scheduler = LinearWarmupScheduler()
        >>> criterion = GPTLMLoss()
        >>>
        >>> model, optimizer, lr_scheduler, dataloader = booster.boost(model, optimizer, lr_scheduler, dataloader)
        >>>
        >>> for epoch in range(max_epochs):
        >>>     for input_ids, attention_mask in dataloader:
        >>>         outputs = model(input_ids, attention_mask)
        >>>         loss = criterion(outputs.logits, input_ids)
        >>>         booster.backward(loss, optimizer)
        >>>         optimizer.step()
        >>>         lr_scheduler.step()
        >>>         optimizer.zero_grad()


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
        # TODO(FrankLeeeee): add plugin control logic
        # if self.plugin is not None and self.plugin.control_accelerator:
        #     ...
        # create acclerator
        self.acceleartor = Accelerator(device)
        self.acceleartor.set_default_device()

        # validate and set precision
        if isinstance(MixedPrecision, str):
            # the user will take the default arguments for amp training
            self.mixed_precision = mixed_precision_factory(mixed_precision)
        elif isinstance(mixed_precision, MixedPrecision):
            # the user can customize the arguments by passing the precision object
            self.mixed_precision = mixed_precision
        else:
            raise ValueError(
                f'Expected the argument mixed_precision to be a string or an instance of Precision, but got {type(mixed_precision)}.'
            )

    def boost(self, model: nn.Module, optimizer: Optimizer, criterion: Callable, lr_scheduler: LRScheduler,
              dataloader: DataLoader) -> List[Union[nn.Module, Optimizer, LRScheduler, DataLoader]]:
        """
        Boost the model, optimizer, criterion, lr_scheduler, and dataloader.

        Args:
            model (nn.Module): The model to be boosted.
            optimizer (Optimizer): The optimizer to be boosted.
            criterion (Callable): The criterion to be boosted.
            lr_scheduler (LRScheduler): The lr_scheduler to be boosted.
            dataloader (DataLoader): The dataloader to be boosted.
        """
        # TODO(FrankLeeeee): add plugin control logic
        # if self.plugin is not None and self.plugin.control_accelerator:
        #     ...
        model = self.acceleartor.configure_model(model)

        # TODO(FrankLeeeee): consider multi-model and multi-optimizer case
        # TODO(lsg): Add plugin control logic
        # e.g.
        # if self.plugin is not None and self.plugin.control_boost:
        #    ...
        # transform model for mixed precision
        model, optimizer, criterion = self.mixed_precision.configure(model, optimizer, criterion)
        return model, optimizer, criterion, lr_scheduler, dataloader

    def backward(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
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
        # TODO: implement this method
        pass

    def save(self,
             obj: Union[nn.Module, Optimizer, LRScheduler],
             path_like: str,
             plan: str = 'torch',
             **kwargs) -> None:
        # TODO: implement this method
        pass

    def load(self,
             obj: Union[nn.Module, Optimizer, LRScheduler],
             path_like: str,
             plan: str = 'torch',
             **kwargs) -> None:
        # TODO: implement this method
        pass
