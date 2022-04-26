from colossalai.core import global_context as gpc
from colossalai.registry import GRADIENT_HANDLER
from ._base_gradient_handler import BaseGradientHandler
from ...context.parallel_mode import ParallelMode
from .utils import bucket_allreduce


@GRADIENT_HANDLER.register_module
class DataParallelGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group.
    A all-reduce collective communication will be operated in 
    :func:`handle_gradient` among a data parallel group.
    For better performance, it bucketizes the gradients of all parameters that are 
    the same type to improve the efficiency of communication.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def handle_gradient(self):
        """A method running a all-reduce operation in a data parallel group.
        """
        # TODO: add memory buffer
        if gpc.data_parallel_size > 1:
            bucket_allreduce(param_list=self._model.parameters(), group=gpc.get_group(ParallelMode.DATA))
