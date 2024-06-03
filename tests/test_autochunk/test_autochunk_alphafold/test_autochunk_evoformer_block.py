from typing import Dict, List, Tuple

import pytest
import torch
import torch.fx

try:
    from fastfold.model.nn.evoformer import EvoformerBlock

    HAS_REPO = True
except:
    HAS_REPO = False

from test_autochunk_alphafold_utils import run_test

from colossalai.autochunk.autochunk_codegen import AUTOCHUNK_AVAILABLE
from colossalai.testing import clear_cache_before_run, parameterize, spawn


def get_model():
    model = (
        EvoformerBlock(
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
        )
        .eval()
        .cuda()
    )
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
        None: [
            (120, 126),
            (225, 244),
            (270, 289),
            (306, 311),
            (70, 106),
            (23, 46),
            (146, 152),
            (187, 193),
            (181, 184),
            (140, 145),
            (162, 163),
            (203, 204),
        ],
        20: [(120, 123), (232, 237), (277, 282), (305, 306)],
        24: [(122, 123)],
    }


@pytest.mark.skipif(
    not (AUTOCHUNK_AVAILABLE and HAS_REPO),
    reason="torch version is lower than 1.12.0",
)
@clear_cache_before_run()
@parameterize("max_memory", [None, 20, 24])
@parameterize("data_args", [(32, 64)])
def test_evoformer_block(data_args, max_memory):
    spawn(
        run_test,
        1,
        data_args=data_args,
        max_memory=max_memory,
        get_model=get_model,
        get_data=get_data,
        get_chunk_target=get_chunk_target,
    )


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
