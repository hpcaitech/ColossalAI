from typing import Callable, Dict, Iterator, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, TensorDataset

import colossalai
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.checkpoint_io import CheckpointIO
from colossalai.interface import OptimizerWrapper
from colossalai.testing import rerun_if_address_is_in_use, spawn


class DPPluginWrapper(DPPluginBase):
    """This is a wrapper class for testing DP plugin initialization and dataloader creation."""

    def configure(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Callable = None,
        dataloader: DataLoader = None,
        lr_scheduler: LRScheduler = None,
    ) -> Tuple[Union[nn.Module, OptimizerWrapper, LRScheduler, DataLoader]]:
        pass

    def control_checkpoint_io(self) -> bool:
        pass

    def control_device(self) -> bool:
        pass

    def control_precision(self) -> bool:
        pass

    def get_checkpoint_io(self) -> CheckpointIO:
        pass

    def support_no_sync(self) -> bool:
        pass

    def supported_devices(self) -> List[str]:
        pass

    def supported_precisions(self) -> List[str]:
        pass

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        pass

    def enable_lora(self, model: nn.Module, pretrained_dir: str, lora_config: Dict) -> nn.Module:
        pass

    def support_lora(self) -> bool:
        pass


def check_dataloader_sharding():
    plugin = DPPluginWrapper()

    # create a custom dataset with 0 to 10
    dataset = TensorDataset(torch.arange(0, 10))
    train_dataloader = plugin.prepare_dataloader(dataset, batch_size=2)

    # get the first batch of data
    batch = next(iter(train_dataloader))[0].cuda()
    is_rank_0 = dist.get_rank() == 0

    if is_rank_0:
        batch_to_compare = batch.clone()
    else:
        batch_to_compare = batch
    # pass to the rank 1 value to rank 0
    dist.broadcast(batch_to_compare, src=1)

    # compare on rank 0
    if is_rank_0:
        assert not torch.equal(
            batch, batch_to_compare
        ), "Same number was found across ranks but expected it to be different"


def run_dist(rank, world_size, port):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_dataloader_sharding()


@rerun_if_address_is_in_use()
def test_dp_plugin_dataloader():
    spawn(run_dist, 2)
