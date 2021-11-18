from colossalai.registry import HOOKS
from colossalai.trainer import BaseHook
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode


@HOOKS.register_module
class TotalBatchsizeHook(BaseHook):
    def __init__(self, trainer, priority: int = 2) -> None:
        super().__init__(trainer, priority)

    def before_train(self):
        total_batch_size = gpc.config.BATCH_SIZE * \
            gpc.config.engine.gradient_accumulation * gpc.get_world_size(ParallelMode.DATA)
        self.logger.info(f'Total batch size = {total_batch_size}', ranks=[0])
