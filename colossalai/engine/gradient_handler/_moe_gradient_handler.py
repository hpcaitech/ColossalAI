from colossalai.core import global_context as gpc, moe_context as moe_env
from colossalai.registry import GRADIENT_HANDLER
from colossalai.utils.moe import get_moe_epsize_param_dict
from ._base_gradient_handler import BaseGradientHandler
from ...context.parallel_mode import ParallelMode
from .utils import bucket_allreduce


@GRADIENT_HANDLER.register_module
class MoeGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group and
    moe model parallel. A all-reduce collective communication will be operated in
    :func:`handle_gradient` among a data parallel group.
    For better performance, it bucketizes the gradients of all parameters that are
    the same type to improve the efficiency of communication.
    """

    def handle_gradient(self):
        """A method running an all-reduce operation in a data parallel group.
        Then running an all-reduce operation for all parameters in experts
        across moe model parallel group
        """
        global_data = gpc.data_parallel_size

        if global_data > 1:
            param_dict = get_moe_epsize_param_dict(self._model)

            # reduce gradients for all parameters in data parallelism
            if 1 in param_dict:
                bucket_allreduce(param_list=param_dict[1], group=gpc.get_group(ParallelMode.DATA))

            for ep_size in param_dict:
                if ep_size != 1 and ep_size != moe_env.world_size:
                    bucket_allreduce(param_list=param_dict[ep_size], group=moe_env.information[ep_size].dp_group)
