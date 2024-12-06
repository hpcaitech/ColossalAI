import tempfile

import torch
from safetensors.torch import load_file

from colossalai.testing import check_state_dict_equal, clear_cache_before_run
from colossalai.utils import get_current_device
from colossalai.utils.safetensors import load_flat, move_and_save, save, save_nested


@clear_cache_before_run()
def test_save_load():
    with tempfile.TemporaryDirectory() as tempdir:
        optimizer_state_dict = {
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

        model_state_dict = {
            "module.weight0": torch.rand((1024, 1024)),
            "module.weight1": torch.rand((1024, 1024)),
            "module.weight2": torch.rand((1024, 1024)),
        }
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
