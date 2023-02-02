from functools import partial
from typing import List, Tuple

import pytest
import torch
import torch.multiprocessing as mp

try:
    from transformers import GPT2Config, GPT2Model
    MODELS = [GPT2Model]
    HAS_REPO = True
except:
    MODELS = []
    HAS_REPO = False

from test_transformer_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE

BATCH_SIZE = 1
SEQ_LENGTH = 512


def get_data(shape: tuple) -> Tuple[List, List]:
    input_ids = torch.zeros(shape, dtype=torch.int64)
    token_type_ids = torch.zeros(shape, dtype=torch.int64)
    attention_mask = torch.ones(shape, dtype=torch.int64)
    meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    concrete_args = {"past_key_values": None}
    sequence = ["input_ids", "past_key_values", "attention_mask", "token_type_ids"]
    return meta_args, concrete_args, sequence


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("shape", [(BATCH_SIZE, SEQ_LENGTH)])
@pytest.mark.parametrize("max_memory", [None, 6, 8])
def test_autochunk_gpt(model, shape, max_memory):
    run_func = partial(
        run_test,
        data=get_data(shape),
        max_memory=max_memory,
        model=model,
        config=GPT2Config(n_embd=96, n_position=shape[1], n_layer=2, n_head=4),
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    run_test(
        rank=0,
        data=get_data((BATCH_SIZE, SEQ_LENGTH)),
        max_memory=None,
        model=GPT2Model,
        config=GPT2Config(n_embd=96, n_position=SEQ_LENGTH, n_layer=2, n_head=4),
        print_code=False,
        print_est_mem=False,
        print_mem=False,
        print_progress=False,
    )
