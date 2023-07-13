import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.amp import convert_to_apex_amp
from colossalai.tensor import ColoTensor, ColoTensorSpec, ComputePattern, ComputeSpec, ProcessGroup, ShardSpec
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer, GeminiDDP, ZeroDDP
from colossalai.zero.gemini import search_chunk_configuration
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import set_seed, tensor_shard_equal
from tests.test_tensor.model.test_gpt2 import init_megatron_spec


def check_param(model: ZeroDDP, torch_model: torch.nn.Module, pg: ProcessGroup):
    zero_dict = model.state_dict(only_rank_0=False)
    torch_dict = torch_model.state_dict()

    for key, value in torch_dict.items():
        # key is 'module.model.PARAMETER', so we truncate it
        key = key[7:]
        assert key in zero_dict, "{} not in ZeRO dictionary.".format(key)
        temp_zero_value = zero_dict[key].to(device=value.device, dtype=value.dtype)
        # debug_print([0], "max range: ", key, torch.max(torch.abs(value - temp_zero_value)))
        assert tensor_shard_equal(value, temp_zero_value, pg.tp_local_rank(), pg.tp_world_size()), \
            "parameter '{}' has problem.".format(key)


def run_fwd_bwd(model, criterion, optimizer, input_ids):
    optimizer.zero_grad()
    logits = model(input_ids)
    logits = logits.float()
    loss = criterion(logits, input_ids)
    optimizer.backward(loss)
    return logits


def init_1d_row_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for n, p in model.named_parameters():
        p.set_process_group(pg)
        if 'weight' in n and 'ln' not in n:
            p.set_tensor_spec(*spec)


def init_1d_col_spec(model, pg: ProcessGroup):
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    for n, p in model.named_parameters():
        p.set_process_group(pg)
        if 'ln' not in n and ('weight' in n or 'bias' in n):
            p.set_tensor_spec(*spec)


@parameterize('placement_policy', ['cuda', 'cpu'])
def run_gpt(placement_policy, tp_init_spec_func=None):
    set_seed(42)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()
    model = model.cuda()
    torch_model = model_builder().cuda()

    for torch_p, p in zip(torch_model.parameters(), model.parameters()):
        torch_p.data.copy_(p.data)

    world_size = torch.distributed.get_world_size()

    # world size, dp = 2, tp =2, construct a hybrid parallelism.
    if world_size == 4:
        pg = ProcessGroup(tp_degree=2)
    else:
        pg = ProcessGroup(tp_degree=world_size)

    if tp_init_spec_func:
        tp_init_spec_func(model, pg)

    dp_world_size = pg.dp_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[dp_world_size]['chunk_size'] = 5000
    config_dict[dp_world_size]['keep_gathered'] = False
    if placement_policy != 'cuda':
        init_device = torch.device('cpu')
    else:
        init_device = None

    model = GeminiDDP(model, init_device, placement_policy, True, False)
    # The same as the following 3 lines
    # chunk_manager = ChunkManager(config_dict, init_device=init_device)
    # gemini_manager = GeminiManager(placement_policy, chunk_manager)
    # model = ZeroDDP(model, gemini_manager, pin_memory=True)

    zero_optim = GeminiAdamOptimizer(model, lr=1e-3, initial_scale=1)
    # The same as the following 2 lines
    # optimizer = HybridAdam(model.parameters(), lr=1e-3)
    # zero_optim = ZeroOptimizer(optimizer, model, initial_scale=1)

    amp_config = dict(opt_level='O2', keep_batchnorm_fp32=False, loss_scale=1)
    torch_optim = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
    torch_model, torch_optim = convert_to_apex_amp(torch_model, torch_optim, amp_config)
    torch_model = DDP(torch_model, device_ids=[pg.rank()], process_group=pg.dp_process_group())

    check_param(model, torch_model, pg)

    model.eval()
    torch_model.eval()

    set_seed(pg.dp_local_rank())
    for i, (input_ids, label) in enumerate(train_dataloader):
        if i > 2:
            break
        input_ids_colo = ColoTensor.from_torch_tensor(input_ids, ColoTensorSpec(pg))
        zero_logits = run_fwd_bwd(model, criterion, zero_optim, input_ids_colo)
        torch_logits = run_fwd_bwd(torch_model, criterion, torch_optim, input_ids)
        assert torch.allclose(zero_logits, torch_logits, rtol=1e-3, atol=1e-2)

        zero_optim.step()
        torch_optim.step()
        check_param(model, torch_model, pg)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    if world_size == 4:
        run_gpt(tp_init_spec_func=init_megatron_spec)
    else:
        run_gpt(tp_init_spec_func=init_1d_col_spec)
        run_gpt(tp_init_spec_func=init_1d_row_spec)


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_gpt(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_gpt(4)
