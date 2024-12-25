import tempfile

import pytest
import torch
from safetensors.torch import load_file

from colossalai.checkpoint_io.utils import create_pinned_state_dict
from colossalai.testing import check_state_dict_equal, clear_cache_before_run
from colossalai.utils import get_current_device
from colossalai.utils.safetensors import load_flat, move_and_save, save, save_nested


def gen_optim_state_dict():
    return {
        "state": {
            0: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.rand((1024, 1024)),
                "exp_avg_sq": torch.rand((1024, 1024)),
            },
            1: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.rand((1024, 1024)),
                "exp_avg_sq": torch.rand((1024, 1024)),
            },
            2: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.rand((1024, 1024)),
                "exp_avg_sq": torch.rand((1024, 1024)),
            },
        },
        "param_groups": [
            {
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "bias_correction": True,
                "params": [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                ],
            }
        ],
    }


def gen_model_state_dict():
    return {
        "module.weight0": torch.rand((1024, 1024)),
        "module.weight1": torch.rand((1024, 1024)),
        "module.weight2": torch.rand((1024, 1024)),
    }


@pytest.mark.parametrize("empty", [True, False])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_create_pin(empty: bool, num_threads: int):
    model_state_dict = gen_model_state_dict()
    model_state_dict_pinned = create_pinned_state_dict(model_state_dict, empty=empty, num_threads=num_threads)
    for k in model_state_dict.keys():
        assert model_state_dict_pinned[k].is_pinned()
        if not empty:
            assert torch.equal(model_state_dict_pinned[k], model_state_dict[k])
    optim_state_dict = gen_optim_state_dict()
    optim_state_dict_pinned = create_pinned_state_dict(optim_state_dict, empty=empty, num_threads=num_threads)
    for k in optim_state_dict.keys():
        if k == "state":
            for idx in optim_state_dict[k].keys():
                for kk in optim_state_dict[k][idx].keys():
                    assert optim_state_dict_pinned[k][idx][kk].is_pinned()
                    if not empty:
                        assert torch.equal(optim_state_dict_pinned[k][idx][kk], optim_state_dict[k][idx][kk])
        else:
            assert optim_state_dict[k] == optim_state_dict_pinned[k]


@clear_cache_before_run()
def test_save_load():
    with tempfile.TemporaryDirectory() as tempdir:
        optimizer_state_dict = gen_optim_state_dict()

        optimizer_saved_path = f"{tempdir}/save_optimizer.safetensors"
        f_writer = save_nested(optimizer_saved_path, optimizer_state_dict)
        f_writer.sync_before_step()
        f_writer.synchronize()
        del f_writer
        load_state_dict = load_flat(optimizer_saved_path)
        check_state_dict_equal(load_state_dict, optimizer_state_dict)

        optimizer_shard_saved_path = f"{tempdir}/save_optimizer_shard.safetensors"
        f_writer = save_nested(optimizer_shard_saved_path, optimizer_state_dict["state"])
        f_writer.sync_before_step()
        f_writer.synchronize()
        del f_writer
        load_state_dict_shard = load_flat(optimizer_shard_saved_path)
        check_state_dict_equal(load_state_dict_shard, optimizer_state_dict["state"])

        model_state_dict = gen_model_state_dict()
        model_saved_path = f"{tempdir}/save_model.safetensors"
        f_writer = save(model_saved_path, model_state_dict)
        f_writer.sync_before_step()
        f_writer.synchronize()
        del f_writer
        load_state_dict = load_file(model_saved_path)
        check_state_dict_equal(model_state_dict, load_state_dict)

        model_state_dict_cuda = {k: v.to(get_current_device()) for k, v in model_state_dict.items()}
        model_state_pinned = {k: v.pin_memory() for k, v in model_state_dict.items()}
        model_saved_path = f"{tempdir}/save_model_cuda.safetensors"
        f_writer = move_and_save(model_saved_path, model_state_dict_cuda, model_state_pinned)
        f_writer.sync_before_step()
        f_writer.synchronize()
        del f_writer
        load_state_dict = load_file(model_saved_path)
        check_state_dict_equal(model_state_dict, load_state_dict)
