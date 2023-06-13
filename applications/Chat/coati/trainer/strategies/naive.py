import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from coati.models.base import get_base_model
from coati.replay_buffer import ReplayBuffer
from coati.models.base import RewardModel
from coati.models.lora import LoraLinear
from coati.replay_buffer import ReplayBuffer
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .base import Strategy


# TODO Move this to a util.py   (Moving to ray.util introduces ringed import)
def get_grad_required_state_dict(model: nn.Module):
    state_dict = OrderedDict()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state_dict[name] = parameter.detach()
    return state_dict


class NaiveStrategy(Strategy):
    """
        Strategy for single GPU. No parallelism is used.
    """

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: optim.Optimizer, **kwargs) -> None:
        optimizer.step()

    def setup_distributed(self) -> None:
        self._try_init_dist(force=False)

    def setup_model(self, model: nn.Module) -> nn.Module:
        return model

    def setup_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        return optimizer

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        return DataLoader(replay_buffer,
                          batch_size=replay_buffer.sample_batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=pin_memory,
                          collate_fn=replay_buffer.collate_fn)

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        state_dict = model.state_dict()
        torch.save(state_dict, path)

    def load_model(self, model: nn.Module, path: str, map_location: Any = None, strict: bool = True) -> None:
        unwrapped_model = self.unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_optimizer(self, optimizer: Optimizer, path: str, only_rank0: bool = False) -> None:
        torch.save(optimizer.state_dict(), path)

    def load_optimizer(self, optimizer: Optimizer, path: str, map_location: Any = None) -> None:
        state_dict = torch.load(path, map_location=map_location)
        optimizer.load_state_dict(state_dict)

    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        unwrapped_model = self.unwrap_model(model)
        assert isinstance(unwrapped_model, PreTrainedModel)
        unwrapped_model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    def get_model_state_dict_shard(self, model: nn.Module, **config):
        # TODO: implement sharding on naive strategy
        model = self.unwrap_model(model)
        if 'requires_grad_only' in config and config['requires_grad_only'] == True:
            state_dict = get_grad_required_state_dict(model)
        else:
            state_dict = model.state_dict()

        if 'shard_size' in config:
            shard_size = config['shard_size']
            accumulate_size = 0
            state_dict_shard = OrderedDict()
            for name, param in state_dict.items():
                state_dict_shard[name] = param
                accumulate_size += param.numel() * param.element_size()
                if accumulate_size >= shard_size:
                    accumulate_size = 0
                    yield state_dict_shard
                    state_dict_shard = OrderedDict()
            if accumulate_size > 0:
                yield state_dict_shard
        else:
            yield state_dict

    def _try_init_dist(self, force: bool = False) -> None:
        try:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            host = os.environ['MASTER_ADDR']
            port = int(os.environ['MASTER_PORT'])
            dist.init_process_group('nccl', init_method=f'tcp://[{host}]:{port}', world_size=world_size, rank=rank)
            torch.cuda.set_device(local_rank)
        except KeyError as e:
            if force:
                raise RuntimeError(
                    f"Could not find {e} in the torch environment, visit https://www.colossalai.org/ for more information on launching with torch"
                )
        except Exception as e:
            if force:
                raise e
