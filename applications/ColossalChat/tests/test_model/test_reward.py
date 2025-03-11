import torch
from coati.distributed.reward.reward_fn import math_reward_fn
from coati.distributed.reward.verifiable_reward import VerifiableReward
from transformers import AutoTokenizer

from colossalai.testing import parameterize
from colossalai.utils import set_seed


@parameterize(
    "test_config",
    [
        {"device": "npu"},
    ],
)
def run_math_reward_fn(test_config):
    device = test_config["device"]
    set_seed(42)
    # init tensor
    input_ids = torch.randint(low=0, high=151644, size=(8, 2304), dtype=torch.int64, device=device)  # [8, 2304]
    gt_answer = torch.randint(low=0, high=151644, size=(8, 128), dtype=torch.int64, device=device)  # [8, 128]
    response_idx = torch.randint(low=256, high=2303, size=(8, 2), dtype=torch.int64, device=device)  # [8, 2]

    # load tokenizer
    # Qwen/Qwen2.5-3B
    tokenizer = AutoTokenizer.from_pretrained("/home/share/data/model/Qwen2.5-3B")

    response_format_tags = {
        "think_start": {"text": "<think>", "num_occur": 1},
        "think_end": {"text": "</think>", "num_occur": 1},
        "answer_start": {"text": "<answer>", "num_occur": 1},
        "answer_end": {"text": "</answer>", "num_occur": 1},
    }
    reward_model = VerifiableReward(reward_fns=[math_reward_fn], tokenizer=tokenizer, tags=response_format_tags)
    reward_model(input_ids, gt_answer, response_idx)


# not in use
def run_gsm8k_reward_fn():
    pass


def test_reward_func():
    run_math_reward_fn()
    # run_gsm8k_reward_fn()


if __name__ == "__main__":
    test_reward_func()
