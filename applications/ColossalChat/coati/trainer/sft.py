"""
SFT trainer
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from coati.trainer.utils import all_reduce_mean
from coati.utils import AccumulativeMeanMeter, save_checkpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin, Plugin
from colossalai.cluster import DistCoordinator

from .base import SLTrainer
from .utils import is_rank_0, to_device


class SFTTrainer(SLTrainer):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        lr_scheduler(_LRScheduler): the lr scheduler to use for training
        max_epochs (int, defaults to 2): the number of epochs to train
        accumulation_steps (int, defaults to 8): the number of steps to accumulate gradients
    """

    def __init__(
        self,
        model,
        booster: Booster,
        optim: Optimizer,
        lr_scheduler: _LRScheduler,
        max_epochs: int = 2,
        plugin: Plugin = None,
        accumulation_steps: int = 8,
        apply_loss_mask: bool = True,
        start_epoch=0,
        save_interval: int = None,
        save_dir: str = None,
        coordinator: Optional[DistCoordinator] = None,
    ) -> None:
        super().__init__(booster, max_epochs, model, optim, plugin, start_epoch=start_epoch)

        self.accumulation_steps = accumulation_steps
        self.scheduler = lr_scheduler
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.coordinator = coordinator
        self.num_train_step = 0
        self.num_eval_step = 0
        self.apply_loss_mask = apply_loss_mask
        self.accumulative_meter = AccumulativeMeanMeter()

    def _before_fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            train_dataloader: the dataloader to use for training
            eval_dataloader: the dataloader to use for evaluation
            log_dir: the directory to save logs
            use_wandb: whether to use wandb for logging
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.writer = None
        if use_wandb and is_rank_0():
            assert log_dir is not None, "log_dir must be provided when use_wandb is True"
            import wandb

            wandb.init(project="Coati-sft", sync_tensorboard=True)
        if log_dir is not None and is_rank_0():
            import os
            import time

            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(log_dir, "sft")
            log_dir = os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epoch: int):
        self.model.train()
        if isinstance(self.plugin, HybridParallelPlugin) and self.plugin.pp_size > 1:
            data_iter = iter(self.train_dataloader)
            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc="Step",
                disable=not (dist.get_rank() == dist.get_world_size() - 1),
            )
            for step in step_bar:
                outputs = self.booster.execute_pipeline(
                    data_iter,
                    self.model,
                    criterion=lambda outputs, inputs: outputs[0],
                    optimizer=self.optimizer,
                    return_loss=True,
                )
                loss = outputs["loss"]

                if self.booster.plugin.stage_manager.is_last_stage():
                    global_loss = all_reduce_mean(loss, self.plugin)
                    if dist.get_rank() == dist.get_world_size() - 1:
                        step_bar.set_postfix({"train/loss": global_loss.item()})

                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            step_bar = trange(
                len(self.train_dataloader) // self.accumulation_steps,
                desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                disable=not is_rank_0(),
            )
            for i, batch in enumerate(self.train_dataloader):
                batch = to_device(batch, torch.cuda.current_device())
                batch_size = batch["input_ids"].size(0)
                outputs = self.model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"] if self.apply_loss_mask else batch["input_ids"],
                )
                loss = outputs.loss

                self.booster.backward(loss=loss, optimizer=self.optimizer)

                loss_mean = all_reduce_mean(tensor=loss)
                self.accumulative_meter.add("loss", loss_mean.to(torch.float16).item())

                # Gradient accumulation
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    step_bar.set_postfix({"train/loss": self.accumulative_meter.get("loss")})
                    if self.writer:
                        self.writer.add_scalar("train/loss", self.accumulative_meter.get("loss"), self.num_train_step)
                        self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.num_train_step)
                    self.num_train_step += 1
                    self.accumulative_meter.reset()
                    step_bar.update()

            # Save checkpoint
            if (
                self.save_dir is not None
                and self.save_interval is not None
                and (self.num_train_step + 1) % self.save_interval == 0
            ):
                save_checkpoint(
                    save_dir=self.save_dir,
                    booster=self.booster,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.scheduler,
                    epoch=epoch,
                    step=self.num_train_step + 1,
                    batch_size=batch_size,
                    coordinator=self.coordinator,
                )
                self.coordinator.print_on_master(
                    f"Saved checkpoint at epoch {epoch} step {self.num_train_step} at folder {self.save_dir}"
                )
        step_bar.close()

    def _eval(self, epoch: int):
        if self.eval_dataloader is None:
            self.coordinator.print_on_master("No eval dataloader is provided, skip evaluation")
            return
        self.accumulative_meter.reset()
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.plugin, HybridParallelPlugin) and self.plugin.pp_size > 1:
                data_iter = iter(self.eval_dataloader)
                step_bar = tqdm(
                    range(len(self.eval_dataloader)),
                    desc="Step",
                    disable=not (dist.get_rank() == dist.get_world_size() - 1),
                )
                for step in step_bar:
                    outputs = self.booster.execute_pipeline(
                        data_iter,
                        self.model,
                        criterion=lambda outputs, inputs: outputs[0],
                        optimizer=self.optimizer,
                        return_loss=True,
                    )
                    loss = outputs["loss"]
                    if self.booster.plugin.stage_manager.is_last_stage():
                        global_loss = all_reduce_mean(loss, self.plugin)
                        if dist.get_rank() == dist.get_world_size() - 1:
                            step_bar.set_postfix({"eval/loss": global_loss.item()})
                            self.accumulative_meter.add("loss", global_loss.item())

                if dist.get_rank() == dist.get_world_size() - 1:
                    loss_mean = self.accumulative_meter.get("loss")
                    msg = "Evaluation Result:\n"
                    for tag in ["loss"]:
                        msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
                    print(msg)
                    if self.save_dir is not None:
                        os.makedirs(self.save_dir, exist_ok=True)
                        with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
                            f.write(msg)
                        step_bar.close()

            else:
                step_bar = trange(
                    len(self.eval_dataloader),
                    desc=f"Epoch {epoch + 1}/{self.max_epochs}",
                    disable=not is_rank_0(),
                )
                for batch in self.eval_dataloader:
                    batch = to_device(batch, torch.cuda.current_device())
                    outputs = self.model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"] if self.apply_loss_mask else batch["input_ids"],
                    )
                    loss_mean = all_reduce_mean(tensor=outputs.loss)
                    self.accumulative_meter.add("loss", loss_mean.item(), count_update=batch["input_ids"].size(0))
                    step_bar.update()

                loss_mean = self.accumulative_meter.get("loss")
                msg = "Evaluation Result:\n"
                for tag in ["loss"]:
                    msg = msg + f"{tag}: {self.accumulative_meter.get(tag)}\n"
                self.coordinator.print_on_master(msg)
                if self.save_dir is not None:
                    os.makedirs(self.save_dir, exist_ok=True)
                    with open(os.path.join(self.save_dir, f"eval_result_epoch{epoch}.txt"), "w") as f:
                        f.write(msg)
                    step_bar.close()
