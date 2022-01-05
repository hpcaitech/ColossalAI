from colossalai.trainer.hooks import BaseHook
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.logging import get_dist_logger
from vilt.modules import heads, objectives, vilt_utils


class CustomizedHook(BaseHook):
    def __init__(self, priority: int = 2) -> None:
        super().__init__(priority)
        self.logger = get_dist_logger()

    def after_train_iter(trainer, output, label, loss):
        total_loss = sum([v for k, v in output.items() if "loss" in k])

    def before_train(self, trainer):
        total_batch_size = gpc.config.BATCH_SIZE * \
            gpc.config.gradient_accumulation * gpc.get_world_size(ParallelMode.DATA)
        self.logger.info(f'Total batch size = {total_batch_size}', ranks=[0])
