import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.context import MOE_CONTEXT
from colossalai.context.moe_context import MOE_CONTEXT
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.engine.gradient_handler._base_gradient_handler import BaseGradientHandler
from colossalai.engine.gradient_handler.utils import bucket_allreduce
from colossalai.nn import CheckpointModule
from colossalai.nn.layer import SparseMLP
from colossalai.registry import GRADIENT_HANDLER
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor
from colossalai.utils.moe import get_moe_epsize_param_dict


class MoeModel(nn.Module):

    def __init__(self, checkpoint: bool = False, expert_parallel: str = "EP"):

        class TestSubModule(CheckpointModule):

            def __init__(self):
                super().__init__(checkpoint)
                self.moe = SparseMLP(num_experts=8,
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


def sync_tp_from_ep(tp_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        tp_model (MoeModule)
        ep_model (MoeModule)
    """
    for (tp_name, tp_param), (ep_name, ep_param) in zip(tp_model.named_parameters(), ep_model.named_parameters()):
        assert tp_name == ep_name
        if not is_moe_tensor(tp_param):
            if assert_grad_flag:
                assert torch.allclose(tp_param, ep_param)
                assert torch.allclose(tp_param.grad, ep_param.grad)
            else:
                tp_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        # get tp param
        tp_dim = [i for i, (d1, d2) in enumerate(zip(tp_param.shape[1:], all_param.shape[1:])) if d1 != d2]
        tp_rank = get_ep_rank(tp_param)
        tp_dim = tp_dim[0] + 1
        tp_slice = [slice(None)] * tp_dim + [
            slice(tp_param.shape[tp_dim] * tp_rank, tp_param.shape[tp_dim] * (tp_rank + 1))
        ]
        new_tp_param = all_param[tuple(tp_slice)]
        if assert_grad_flag:
            new_grad = all_grad[tuple(tp_slice)]
        if assert_grad_flag:
            assert torch.allclose(tp_param, new_tp_param)
            assert torch.allclose(tp_param.grad, new_grad)
        else:
            tp_param.data.copy_(new_tp_param.data)
