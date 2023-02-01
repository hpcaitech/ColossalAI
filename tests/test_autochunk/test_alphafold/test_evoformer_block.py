from functools import partial
from typing import Dict, List, Tuple

import pytest
import torch
import torch.fx
import torch.multiprocessing as mp

try:
    from fastfold.model.nn.evoformer import EvoformerBlock
    HAS_REPO = True
except:
    HAS_REPO = False

from test_alphafold_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE


def get_model():
    model = EvoformerBlock(
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
    concrete_args = [("chunk_size", None), ("_mask_trans", True)]
    return meta_args, concrete_args


def get_chunk_target() -> Dict:
    return {
        None: [(120, 123), (222, 237), (269, 289), (305, 311), (100, 105), (146, 152), (187, 193), (241, 242),
               (25, 50)],
        20: [(120, 123), (232, 237), (277, 282), (305, 306), (100, 101), (34, 39)],
        24: [(120, 123)],
    }


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@pytest.mark.parametrize("max_memory", [None, 20, 24])
@pytest.mark.parametrize("data_args", [(32, 64)])    # (msa_len, pair_len)
def test_evoformer_block(data_args, max_memory):
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
        max_memory=24,
        get_model=get_model,
        get_data=get_data,
        get_chunk_target=get_chunk_target,
        print_code=False,
        print_mem=False,
        print_est_mem=False,
        print_progress=False,
    )
