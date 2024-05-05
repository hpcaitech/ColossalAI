from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.pytorch_utils import Conv1D

try:
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_model

    NO_CODEGEN = False
except:
    NO_CODEGEN = True

from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use, run_on_environment_flag, spawn

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
    launch(rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    model = GPT2MLPWithCkpt(intermediate_size=4 * HIDDEN_SIZE, hidden_size=HIDDEN_SIZE)
    torch.rand(1, 64, HIDDEN_SIZE)
    input_sample = {
        "hidden_states": torch.rand(1, 64, HIDDEN_SIZE).to("meta"),
    }
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = (2, 2)
    # [[0, 1]
    #  [2, 3]]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    gm = initialize_model(model, input_sample, device_mesh)
    code = gm.module.graph.python_code("self").src
    assert (
        "runtime_comm_spec_apply_1 = colossalai_auto_parallel_passes_runtime_apply_pass_runtime_comm_spec_apply(linear_1, comm_actions_dict, 12, 'linear_1')"
        in code
    )
    assert (
        "view_3 = torch.utils.checkpoint.checkpoint(self.checkpoint_0, view_1, comm_actions_dict, use_reentrant=False)"
        in code
    )


@run_on_environment_flag(name="AUTO_PARALLEL")
@pytest.mark.skipif(NO_CODEGEN, reason="No codegen found")
@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_mlp_layer():
    spawn(check_act_ckpt, 4)


if __name__ == "__main__":
    test_mlp_layer()
