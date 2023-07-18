import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils.cuda import get_current_device
from colossalai.zero import ColoInitContext, ZeroDDP, ZeroOptimizer
from colossalai.zero.gemini.chunk import ChunkManager, search_chunk_configuration
from colossalai.zero.gemini.gemini_mgr import GeminiManager
from tests.components_to_test.registry import non_distributed_component_funcs
from tests.test_tensor.common_utils import debug_print, set_seed


@parameterize('placement_policy', ['cuda', 'cpu', 'auto'])
@parameterize('keep_gathered', [True, False])
def exam_zero_optim_state_dict(placement_policy, keep_gathered):
    set_seed(431)
    get_components_func = non_distributed_component_funcs.get_callable('gpt2')
    model_builder, train_dataloader, test_dataloader, optimizer_class, criterion = get_components_func()

    with ColoInitContext(device=get_current_device()):
        model = model_builder()

    set_seed(451)
    torch_model = model_builder()    # get a different model

    world_size = torch.distributed.get_world_size()
    config_dict, *_ = search_chunk_configuration(model, search_range_m=1, search_interval=100)
    config_dict[world_size]['chunk_size'] = 5000
    config_dict[world_size]['keep_gathered'] = keep_gathered

    if placement_policy != 'cuda':
        init_device = torch.device('cpu')
    else:
        init_device = None
    chunk_manager = ChunkManager(config_dict, init_device=init_device)
    gemini_manager = GeminiManager(placement_policy, chunk_manager)
    model = ZeroDDP(model, gemini_manager, pin_memory=True)

    optimizer = HybridAdam(model.parameters())
    optim = ZeroOptimizer(optimizer, model, initial_scale=32)    # initialize the link between chunk16 and chunk32

    set_seed(dist.get_rank() * 3 + 128)
    model.train()
    for i, (input_ids, label) in enumerate(train_dataloader):
        if i > 0:
            break
        optim.zero_grad()
        logits = model(input_ids)
        logits = logits.float()
        loss = criterion(logits, input_ids)
        optim.backward(loss)
        optim.step()

    optim_state_dict = optim.state_dict()
    optim.load_state_dict(optim_state_dict)
    new_state = optim.state_dict()['state']
    org_state = optim_state_dict['state']

    for k, v in org_state.items():
        w = new_state[k]
        for n, m in v.items():
            if isinstance(m, torch.Tensor):
                o = w[n]
                assert torch.equal(m, o)
            else:
                assert m == w[n]


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    exam_zero_optim_state_dict()


@pytest.mark.dist
@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_zero_optim(world_size):
    spawn(run_dist, world_size)


if __name__ == '__main__':
    test_zero_optim(1)
