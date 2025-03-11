import copy

import torch
from coati.distributed.loss import PolicyLoss
from torch.testing import assert_close

from colossalai.testing import parameterize
from colossalai.utils import set_seed


@parameterize(
    "test_config",
    [
        {
            "precision": torch.bfloat16,
            "device": "npu",
        },
    ],
)
def run_policy_loss_fn(test_config):
    dtype = test_config["precision"]
    device = test_config["device"]
    set_seed(42)
    policy_loss_fn = PolicyLoss()

    ############
    # init npu tensor
    ############
    action_log_probs = torch.rand(8, 2048, dtype=dtype, device=device)  # float [8, 2048]
    old_action_log_probs = torch.rand(8, 2048, dtype=dtype, device=device)  # float [8, 2048]
    advantages = torch.rand(8, dtype=dtype, device=device)  # float [8]
    per_token_kl = torch.rand(8, 2048, dtype=dtype, device=device)  # float [8, 2048]
    action_mask = torch.randint(
        low=0, high=2, size=(8, 2048), dtype=torch.int32, device=device
    )  # torch.int32 [8, 2048] in range(0,1)

    loss, skip_update, _ = policy_loss_fn(
        action_log_probs,
        old_action_log_probs,
        advantages.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
        per_token_kl,
        action_mask,
    )

    ############
    # init cpu tensor
    ############
    action_log_probs_cpu = copy.deepcopy(action_log_probs.cpu())
    old_action_log_probs_cpu = copy.deepcopy(old_action_log_probs.cpu())
    advantages_cpu = copy.deepcopy(advantages.cpu())
    per_token_kl_cpu = copy.deepcopy(per_token_kl.cpu())
    action_mask_cpu = copy.deepcopy(action_mask.cpu())

    loss_cpu, skip_update_cpu, _ = policy_loss_fn(
        action_log_probs_cpu,
        old_action_log_probs_cpu,
        advantages_cpu.unsqueeze(dim=-1).repeat_interleave(action_log_probs.size(-1), dim=-1),
        per_token_kl_cpu,
        action_mask_cpu,
    )

    # assert close
    assert_close(
        loss.to("cpu"),
        loss_cpu,
        rtol=5e-4,
        atol=5e-4,
        # msg=f"NPU/CPU {test_config['precision']} not close"
    )


def test_loss_func():
    run_policy_loss_fn()


if __name__ == "__main__":
    test_loss_func()
