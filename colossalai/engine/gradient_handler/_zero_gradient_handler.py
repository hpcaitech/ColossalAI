from colossalai.registry import GRADIENT_HANDLER
from ._base_gradient_handler import BaseGradientHandler


@GRADIENT_HANDLER.register_module
class ZeROGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group.
    A all-reduce collective communication will be operated in
    :func:`handle_gradient` among a data parallel group.
    This class is specialized with ZeRO optimization.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def handle_gradient(self):
        """A method running a all-reduce operation in a data parallel group.
        """
        self._optimizer.sync_grad()
