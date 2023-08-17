import torch.nn as nn

from colossalai.context import MOE_CONTEXT
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.gradient_handler._base_gradient_handler import BaseGradientHandler
from colossalai.engine.gradient_handler.utils import bucket_allreduce
from colossalai.nn import CheckpointModule
from colossalai.nn.layer import MoeModule
from colossalai.registry import GRADIENT_HANDLER
from colossalai.utils.moe import get_moe_epsize_param_dict


class MoeModel(nn.Module):

    def __init__(self, checkpoint: bool = False, expert_parallel: str = "EP"):

        class TestSubModule(CheckpointModule):

            def __init__(self):
                super().__init__(checkpoint)
                self.moe = MoeModule(num_experts=8,
                                     use_residual=False,
                                     expert_parallel=expert_parallel,
                                     hidden_size=16,
                                     intermediate_size=32)
                self.proj = nn.Linear(16, 4)

            def _forward(self, x):
                x, y = self.moe(x)
                x = self.proj(x)
                return x, y

        super().__init__()
        self.test_embed = nn.Linear(4, 16)
        self.test_transform = TestSubModule()

    def forward(self, x):
        MOE_CONTEXT.reset_loss()

        x = self.test_embed(x)
        x, y = self.test_transform(x)

        MOE_CONTEXT.add_loss(y)
        return x


@GRADIENT_HANDLER.register_module
class MoeGradientHandler(BaseGradientHandler):
    """A helper class to handle all-reduce operations in a data parallel group and
    moe model parallel. A all-reduce collective communication will be operated in
    :func:`handle_gradient` among a data parallel group.
    For better performance, it bucketizes the gradients of all parameters that are
    the same type to improve the efficiency of communication.

    Args:
        model (Module): Model where the gradients accumulate.
        optimizer (Optimizer): Optimizer for updating the parameters.
    """

    def __init__(self, model, optimizer=None):
        super().__init__(model, optimizer)

    def handle_gradient(self):
        """A method running an all-reduce operation in a data parallel group.
        Then running an all-reduce operation for all parameters in experts
        across moe model parallel group
        """
        global_data = gpc.data_parallel_size

        if global_data > 1:
            epsize_param_dict = get_moe_epsize_param_dict(self._model)

            # epsize is 1, indicating the params are replicated among processes in data parallelism
            # use the ParallelMode.DATA to get data parallel group
            # reduce gradients for all parameters in data parallelism
            if 1 in epsize_param_dict:
                bucket_allreduce(param_list=epsize_param_dict[1], group=gpc.get_group(ParallelMode.DATA))

            for ep_size in epsize_param_dict:
                if ep_size != 1 and ep_size != MOE_CONTEXT.world_size:
                    bucket_allreduce(param_list=epsize_param_dict[ep_size],
                                     group=MOE_CONTEXT.parallel_info_dict[ep_size].dp_group)
