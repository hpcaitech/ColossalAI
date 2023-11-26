import os
import time

import wandb
from torch.utils.tensorboard import SummaryWriter


class WandbLog:
    @classmethod
    def init_wandb(cls, project, notes=None, name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), config=None):
        wandb.init(project=project, notes=notes, name=name, config=config)

    @classmethod
    def log(cls, result, model=None, gradient=None):
        wandb.log(result)

        if model:
            wandb.watch(model)

        if gradient:
            wandb.watch(gradient)


class TensorboardLog:
    def __init__(self, location, name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), config=None):
        if not os.path.exists(location):
            os.mkdir(location)
        self.writer = SummaryWriter(location, comment=name)

    def log_train(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f"{k}/train", v, step)

    def log_eval(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f"{k}/eval", v, step)

    def log_zeroshot(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f"{k}_acc/eval", v, step)
