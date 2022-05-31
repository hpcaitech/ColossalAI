import pytest
import colossalai
from colossalai.context.parallel_mode import ParallelMode
import torch.multiprocessing as mp
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils import free_port
from colossalai.utils import ColoInitContext
from colossalai.tensor import TensorSpec, ComputePattern, ParallelAction, DistSpecManager, distspec, ColoParameter, ChunkManager
from colossalai.core import global_context as gpc
from functools import partial
from _utils import tensor_equal, tensor_shard_equal, set_seed
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from colossalai.nn.parallel import ColoDDP, ColoDDPV2
from colossalai.testing import parameterize


def check_param_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        if p.storage().size() > 0:
            assert tensor_equal(torch_p, p.float()), f'{torch_p} vs {p}'


def check_grad_equal(model, torch_model):
    for p, torch_p in zip(model.parameters(), torch_model.parameters()):
        if p.grad is not None:
            assert tensor_equal(torch_p.grad, p.grad.float())


@parameterize('use_chunk', [False, True])
@parameterize('use_zero', [False, True])
def run_gpt(use_chunk, use_zero):
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=True)
    model = model.cuda()
    torch_model = model_builder().cuda()
    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p)
    model = model.half()
    chunk_size = 38 * 1024**2 if use_chunk else None
    chunk_manager = ChunkManager(chunk_size, enable_distributed_storage=use_zero)
    model = ColoDDPV2(model, chunk_manager)
    torch_model = DDP(torch_model, device_ids=[gpc.get_global_rank()], process_group=gpc.get_group(ParallelMode.DATA))
    print(chunk_manager)
    check_param_equal(model, torch_model)
    model.train()
    torch_model.train()
    set_seed(gpc.get_local_rank(ParallelMode.DATA))
    for i, (input_ids, attn_mask) in enumerate(train_dataloader):
        logits = model(input_ids, attn_mask)
        torch_logits = torch_model(input_ids, attn_mask)
        assert tensor_equal(torch_logits, logits.float())
        loss = criterion(logits, input_ids)
        torch_loss = criterion(torch_logits, input_ids)
        model.backward(loss)
        torch_loss.backward()
        check_grad_equal(model, torch_model)
        break


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_gpt(4)
