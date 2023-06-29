import os
import random
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from coati.replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.booster.plugin.torch_ddp_plugin import TorchDDPModel

from .base import Strategy
from .sampler import DistributedSampler


# TODO Move this to a util.py   (Moving to ray.util introduces ringed import)
def get_grad_required_state_dict(model: nn.Module):
    state_dict = OrderedDict()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            state_dict[name] = parameter.detach()
    return state_dict


class DDPStrategy(Strategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def __init__(self,
                 seed: int = 42,
                 plugin_initializer: Callable = TorchDDPPlugin
                 ) -> None:
        self.seed = seed
        super().__init__(plugin_initializer)

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

    def _post_init(self) -> None:
        assert isinstance(self.plugin, TorchDDPPlugin), \
            f'{type(self).__name__}\'s plugin is not initialized properly.'

    def setup_distributed(self) -> None:
        self._try_init_dist(force=True)
        self.set_seed(self.seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def setup_dataloader(self, replay_buffer: ReplayBuffer, pin_memory: bool = False) -> DataLoader:
        return self.plugin.prepare_dataloader(replay_buffer,
                                              batch_size=replay_buffer.sample_batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              pin_memory=pin_memory,
                                              collate_fn=replay_buffer.collate_fn)

    def setup_sampler(self, dataset) -> DistributedSampler:
        # FIXME(cwher): this is only invoked in train_on_ray, not tested after adapt Boost API.
        return DistributedSampler(dataset, dist.get_world_size(), dist.get_rank())

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        assert isinstance(model, TorchDDPModel), "model is not wrapped by TorchDDPModel."
        return model.unwrap()

    def save_pretrained(self,
                        model: nn.Module,
                        path: str,
                        only_rank0: bool = True,
                        tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
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
