import os
from typing import Optional, Tuple

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from deepspeed import DeepSpeedEngine

from .ddp import DDPStrategy


class DeepspeedStrategy(DDPStrategy):
    """
        The strategy for training with Deepspeed.


    """

    def __init__(
        self,
        stage: int = 0,
        seed: int = 42,
        fp16: bool = True,
        train_batch_size: int = 8,
        lr: float = 5e-6,
        optimizer: str = "Adam",
        offload_optimizer: str = 'none',    # "cpu" or "nvme", "nvme" is available only with ZeRO stage 3
        offload_param: str = 'none'    # "cpu" or "nvme", "nvme" is available only with ZeRO stage 3
    ) -> None:
        super().__init__(seed)
        self.config = {
            "train_micro_batch_size_per_gpu": train_batch_size,
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": lr,
                }
            },
            "fp16": {
                "enabled": fp16,
            },
            "zero_optimization": {
                "stage": stage,
                "offload_optimizer": {
                    "device": offload_optimizer,
                },
                "offload_param": {
                    "device": offload_param,
                },
            }
        }

    def setup_distributed(self) -> None:
        deepspeed.init_distributed(dist_backend='nccl')

    def setup_model_and_optimizer(self, model: nn.Module):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        deepspeed_engine, deepspeed_optimizer, _, _ = deepspeed.initialize(
            config=self.config,
            model=model,
            model_parameters=model_parameters,
            dist_init_required=False,
        )
        return deepspeed_engine, deepspeed_optimizer

    def setup_model(self, model: nn.Module) -> DeepSpeedEngine:
        deepspeed_engine, _, _, _ = deepspeed.initialize(
            config=self.config,
            model=model,
        )
        return deepspeed_engine

    def backward(self, loss: torch.Tensor, deepspeed_engine: DeepSpeedEngine, **kwargs) -> None:
        deepspeed_engine.backward(loss)

    def optimizer_step(self, deepspeed_engine: DeepSpeedEngine, **kwargs) -> None:
        deepspeed_engine.step()

    def save_model(self, deepspeed_engine: DeepSpeedEngine, path: str, only_rank0: bool = False) -> None:
        if only_rank0 and dist.get_rank() != 0:
            return
        deepspeed_engine.save_checkpoint(path)
