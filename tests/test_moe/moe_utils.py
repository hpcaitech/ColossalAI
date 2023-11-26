import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.legacy.engine.gradient_handler._base_gradient_handler import BaseGradientHandler
from colossalai.legacy.engine.gradient_handler.utils import bucket_allreduce
from colossalai.legacy.registry import GRADIENT_HANDLER
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import get_moe_epsize_param_dict


class MoeModel(nn.Module):
    def __init__(self, enable_load_balance: bool = False):
        class TestSubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = SparseMLP(
                    num_experts=8, hidden_size=16, intermediate_size=32, enable_load_balance=enable_load_balance
                )
                self.proj = nn.Linear(16, 4)

            def forward(self, x):
                x = self.moe(x)
                x = self.proj(x)
                return x

        super().__init__()
        self.test_embed = nn.Linear(4, 16)
        self.test_transform = TestSubModule()

    def forward(self, x):
        MOE_MANAGER.reset_loss()

        x = self.test_embed(x)
        x = self.test_transform(x)

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
        if dist.get_world_size() > 1:
            epsize_param_dict = get_moe_epsize_param_dict(self._model)

            # epsize is 1, indicating the params are replicated among processes in data parallelism
            # use the ParallelMode.DATA to get data parallel group
            # reduce gradients for all parameters in data parallelism
            if 1 in epsize_param_dict:
                bucket_allreduce(param_list=epsize_param_dict[1])

            for ep_size in epsize_param_dict:
                if ep_size != 1 and ep_size != MOE_MANAGER.world_size:
                    bucket_allreduce(
                        param_list=epsize_param_dict[ep_size], group=MOE_MANAGER.parallel_info_dict[ep_size].dp_group
                    )


def assert_not_equal_in_group(tensor, process_group=None):
    # all gather tensors from different ranks
    world_size = dist.get_world_size(process_group)
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=process_group)

    # check if they are equal one by one
    for i in range(world_size - 1):
        a = tensor_list[i]
        b = tensor_list[i + 1]
        assert not torch.allclose(a, b), \
            (f"expected tensors on rank {i} and {i + 1} not to be equal "
             f"but they are, {a} vs {b}")
