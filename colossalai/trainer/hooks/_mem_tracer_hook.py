from colossalai.registry import HOOKS
from torch import Tensor
from colossalai.trainer.hooks import BaseHook
from colossalai.gemini.memory_tracer import AsyncMemoryMonitor


@HOOKS.register_module
class MemTraceHook(BaseHook):
    """Save memory stats and pass it to states
    This hook is used to record memory usage info, and pass to trainer.states
    You can use it as other trainer hook and fetch data from trainer.states['metrics][mode]
    """

    def __init__(
        self,
        priority: int = 0,
    ) -> None:
        super().__init__(priority=priority)
        self._memory_monitor = AsyncMemoryMonitor()

    def after_hook_is_attached(self, trainer):
        # Initialize the data
        trainer.states['metrics']['train'] = self._memory_monitor.state_dict
        trainer.states['metrics']['test'] = self._memory_monitor.state_dict

    def before_train_iter(self, trainer):
        self._memory_monitor.start()
        return super().before_train_iter(trainer)

    def after_train_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        self._memory_monitor.finish()
        trainer.states['metrics']['train'] = self._memory_monitor.state_dict
        trainer.states['metrics']['test'] = self._memory_monitor.state_dict
        return super().after_train_iter(trainer, output, label, loss)

    def before_test_iter(self, trainer):
        self._memory_monitor.start()
        return super().before_test(trainer)

    def after_test_iter(self, trainer, output: Tensor, label: Tensor, loss: Tensor):
        self._memory_monitor.finish()
        trainer.states['metrics']['train'] = self._memory_monitor.state_dict
        trainer.states['metrics']['test'] = self._memory_monitor.state_dict
        return super().after_test_iter(trainer, output, label, loss)
