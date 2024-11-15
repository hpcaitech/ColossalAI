import tempfile
from copy import deepcopy

import torch

from colossalai.utils.safetensors import load_flat, save_nested

try:
    from tensornvme.async_file_io import AsyncFileWriter
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install tensornvme to use NVMeOptimizer")

from colossalai.testing import check_state_dict_equal


def test_save_load():
    with tempfile.TemporaryDirectory() as tempdir:
        optimizer_state_dict = {
            0: {"step": torch.tensor(1.0), "exp_avg": torch.rand((1024, 1024)), "exp_avg_sq": torch.rand((1024, 1024))},
            1: {"step": torch.tensor(1.0), "exp_avg": torch.rand((1024, 1024)), "exp_avg_sq": torch.rand((1024, 1024))},
            2: {"step": torch.tensor(1.0), "exp_avg": torch.rand((1024, 1024)), "exp_avg_sq": torch.rand((1024, 1024))},
        }
        group_dict = {"param_groups": [0, 1, 2]}
        metadata = deepcopy(group_dict)
        optimizer_saved_path = f"{tempdir}/save_optimizer.safetensors"
        f_writer = AsyncFileWriter(fp=open(optimizer_saved_path, "wb"), n_entries=191, backend="pthread")

        save_nested(f_writer, optimizer_state_dict, metadata)
        f_writer.sync_before_step()
        f_writer.synchronize()
        f_writer.fp.close()

        load_state_dict = load_flat(optimizer_saved_path)
        state_dict = load_state_dict["state"]
        group = {"param_groups": load_state_dict["param_groups"]}
        check_state_dict_equal(optimizer_state_dict, state_dict)
        check_state_dict_equal(group_dict, group)

        model_state_dict = {
            "module.weight0": torch.rand((1024, 1024)),
            "module.weight1": torch.rand((1024, 1024)),
            "module.weight2": torch.rand((1024, 1024)),
        }
        model_saved_path = f"{tempdir}/save_model.safetensors"
        f_writer = AsyncFileWriter(fp=open(model_saved_path, "wb"), n_entries=191, backend="pthread")
        save_nested(f_writer, model_state_dict)
        f_writer.sync_before_step()
        f_writer.synchronize()
        f_writer.fp.close()

        load_state_dict = load_flat(model_saved_path)
        check_state_dict_equal(model_state_dict, load_state_dict)
