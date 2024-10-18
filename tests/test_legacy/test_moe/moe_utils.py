import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.booster.plugin.low_level_zero_plugin import LowLevelZeroModel
from colossalai.legacy.engine.gradient_handler._base_gradient_handler import BaseGradientHandler
from colossalai.legacy.engine.gradient_handler.utils import bucket_allreduce
from colossalai.legacy.moe.manager import MOE_MANAGER
from colossalai.legacy.moe.utils import get_moe_epsize_param_dict
from colossalai.legacy.registry import GRADIENT_HANDLER
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_size, set_moe_tensor_ep_group


def delete_moe_info(model):
    for _, param in model.named_parameters():
        if hasattr(param, "ep_group"):
            delattr(param, "ep_group")


class MoeModel(nn.Module):
    def __init__(self, ep_group: ProcessGroup = None):
        super().__init__()
        self.test_embed = nn.Linear(4, 16, bias=False)
        self.w1 = torch.nn.Parameter(torch.randn(16, 8))
        if ep_group:
            set_moe_tensor_ep_group(self.w1, ep_group)

    def forward(self, x):
        x = self.test_embed(x)
        x = torch.matmul(x, self.w1)

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
        assert not torch.allclose(a, b), (
            f"expected tensors on rank {i} and {i + 1} not to be equal " f"but they are, {a} vs {b}"
        )


def run_fwd_bwd(model, data, label, criterion, optimizer, enable_autocast=False):
    model.train()
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.float()

    if isinstance(model, LowLevelZeroModel):
        optimizer.backward(loss)
    else:
        loss.backward()
    return y


def sync_local_from_ep(local_model, ep_model, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        local_model (MoeModule)
        ep_model (MoeModule)
    """
    for (local_name, local_param), (ep_name, ep_param) in zip(
        local_model.named_parameters(), ep_model.named_parameters()
    ):
        if "experts" not in local_name:
            if assert_grad_flag:
                assert torch.allclose(local_param, ep_param), f"local_param: {local_param}, ep_param: {ep_param}"
                assert torch.allclose(local_param.grad, ep_param.grad)
            else:
                local_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        if assert_grad_flag:
            assert torch.allclose(local_param, all_param)
            assert torch.allclose(local_param.grad, all_grad)
        else:
            local_param.data.copy_(all_param.data)
