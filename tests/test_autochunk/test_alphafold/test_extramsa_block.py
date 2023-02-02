from functools import partial
from typing import Dict, List, Tuple

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

try:
    from fastfold.model.nn.evoformer import ExtraMSABlock
    HAS_REPO = True
except:
    HAS_REPO = False
from test_alphafold_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE


def get_model():
    model = ExtraMSABlock(
        c_m=256,
        c_z=128,
        c_hidden_msa_att=32,
        c_hidden_opm=32,
        c_hidden_mul=128,
        c_hidden_pair_att=32,
        no_heads_msa=8,
        no_heads_pair=4,
        transition_n=4,
        msa_dropout=0.15,
        pair_dropout=0.15,
        inf=1e4,
        eps=1e-4,
        ckpt=False,
        is_multimer=False,
    ).eval().cuda()
    return model


def get_data(msa_len: int, pair_len: int) -> Tuple[List, List]:
    node = torch.randn(1, msa_len, pair_len, 256).cuda()
    node_mask = torch.randn(1, msa_len, pair_len).cuda()
    pair = torch.randn(1, pair_len, pair_len, 128).cuda()
    pair_mask = torch.randn(1, pair_len, pair_len).cuda()

    meta_args = [
        ("m", node),
        ("z", pair),
        ("msa_mask", node_mask),
        ("pair_mask", pair_mask),
    ]
    concrete_args = [("chunk_size", None), ("_chunk_logits", 1024)]
    return meta_args, concrete_args


def get_chunk_target() -> Dict:
    return {
        None: [(128, 131), (230, 245), (277, 297), (313, 319), (108, 113), (154, 160), (195, 201), (249, 250),
               (36, 46)],
        20: [(128, 131), (240, 245), (285, 290), (313, 314), (108, 109), (41, 46)],
        24: [(128, 131)],
    }


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@pytest.mark.parametrize("max_memory", [None, 20, 24])
@pytest.mark.parametrize("data_args", [(32, 64)])    # (msa_len, pair_len)
def test_extramsa_block(data_args, max_memory):
    run_func = partial(
        run_test,
        data_args=data_args,
        max_memory=max_memory,
        get_model=get_model,
        get_data=get_data,
        get_chunk_target=get_chunk_target,
    )
    mp.spawn(run_func, nprocs=1)


if __name__ == "__main__":
    run_test(
        rank=0,
        data_args=(32, 64),
        max_memory=None,
        get_model=get_model,
        get_data=get_data,
        get_chunk_target=get_chunk_target,
        print_code=False,
        print_mem=False,
        print_progress=False,
    )
