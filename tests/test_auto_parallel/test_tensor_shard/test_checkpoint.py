from functools import partial
from typing import Optional, Tuple, Union

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.pytorch_utils import Conv1D

from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
from colossalai.device.device_mesh import DeviceMesh
from colossalai.fx.graph_module import ColoGraphModule
from colossalai.fx.tracer import ColoTracer
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.testing.pytest_wrapper import run_on_environment_flag
from colossalai.utils import free_port

HIDDEN_SIZE = 16


class GPT2MLPWithCkpt(nn.Module):

    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        embed_dim = hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = torch.nn.ReLU()

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = checkpoint(self.c_proj, hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


def check_act_ckpt(rank, world_size, port):
    disable_existing_loggers()
    launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    model = GPT2MLPWithCkpt(intermediate_size=4 * HIDDEN_SIZE, hidden_size=HIDDEN_SIZE)
    input_sample = {
        'hidden_states': torch.rand(1, 64, HIDDEN_SIZE).to('meta'),
    }
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    gm = initialize_model(model, input_sample, device_mesh)
    code = gm.module.graph.python_code('self').src
    assert "runtime_comm_spec_apply_1 = colossalai_auto_parallel_passes_runtime_apply_pass_runtime_comm_spec_apply(linear_1, comm_actions_dict, 12, 'linear_1')" in code
    assert "view_3 = colossalai.utils.activation_checkpoint.checkpoint(self.checkpoint_0, False, view_1, comm_actions_dict, use_reentrant=True)" in code


@run_on_environment_flag(name='AUTO_PARALLEL')
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_mlp_layer():
    world_size = 4
    run_func = partial(check_act_ckpt, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_mlp_layer()
