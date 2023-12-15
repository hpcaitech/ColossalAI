import copy

import torch
from colossal_moe.models.mixtral_layer import MixtralSparseMLP
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


class Config:
    def __init__(self, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, hidden_act):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = hidden_act


def test_moe_layer():
    config = Config(hidden_size=4, intermediate_size=8, num_local_experts=32, num_experts_per_tok=2, hidden_act="silu")
    mistral_moe = MixtralSparseMoeBlock(config).cuda()
    colossal_moe = MixtralSparseMLP.from_native_module(copy.deepcopy(mistral_moe)).cuda()

    data = torch.randn(2, 8, 4).cuda()
    mistral_output = mistral_moe(data)[0]
    colossal_output = colossal_moe(data)[0]
    assert torch.allclose(
        mistral_output, colossal_output
    ), f"mistral_output: {mistral_output}\ncolossal_output: {colossal_output}"


if __name__ == "__main__":
    test_moe_layer()
