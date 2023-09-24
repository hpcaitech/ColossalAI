from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.fx import is_compatible_with_meta
from colossalai.lazy.lazy_init import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def run_fn(init_method, model_fn, data_gen_fn, output_transform_fn) -> Optional[str]:
    try:
        if init_method == "lazy":
            ctx = LazyInitContext()
        else:
            ctx = nullcontext()
        plugin = HybridParallelPlugin(tp_size=2, pp_size=2, num_microbatches=4, precision="bf16")
        booster = Booster(plugin=plugin)
        with ctx:
            model = model_fn()
        optimizer = HybridAdam(model.parameters(), lr=1e-3)
        criterion = lambda x: x.mean()
        data = data_gen_fn()

        data = {
            k: v.to("cuda").repeat(4, 1) if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v
            for k, v in data.items()
        }

        model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

        data_iter = iter([data])

        def _criterion(outputs, inputs):
            outputs = output_transform_fn(outputs)
            output_key = list(outputs.keys())[0]
            loss = criterion(outputs[output_key])
            return loss

        booster.execute_pipeline(data_iter, model, _criterion, optimizer, return_loss=True, return_outputs=False)
        optimizer.step()

    except Exception as e:
        return repr(e)


@parameterize("init_method", ["none", "lazy"])
def check_3d_plugin(init_method: str = "none", early_stop: bool = True):
    """check gemini plugin over model zoo

    Args:
        early_stop (bool, optional): Whether to stop when getting the first error. Defaults to True.
    """
    is_support_meta = is_compatible_with_meta()
    if not is_support_meta and init_method == "lazy":
        return

    passed_models = []
    failed_info = {}  # (model_name, error) pair

    # TODO(ver217): add more models
    for name, (model_fn, data_gen_fn, output_transform_fn, _, _) in model_zoo.get_sub_registry(
        "transformers_llama_for_casual_lm"
    ).items():
        err = run_fn(init_method, model_fn, data_gen_fn, output_transform_fn)
        torch.cuda.empty_cache()

        if err is None:
            passed_models.append(name)
        else:
            failed_info[name] = err
            if early_stop:
                break

    if dist.get_rank() == 0:
        print(f"Init method: {init_method}")
        print(f"Passed models({len(passed_models)}): {passed_models}\n\n")
        print(f"Failed models({len(failed_info)}): {list(failed_info.keys())}\n\n")
    assert len(failed_info) == 0, "\n".join([f"{k}: {v}" for k, v in failed_info.items()])


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(config=dict(), rank=rank, world_size=world_size, port=port, host="localhost")
    check_3d_plugin(early_stop=early_stop)


@rerun_if_address_is_in_use()
def test_gemini_plugin(early_stop: bool = True):
    spawn(run_dist, 4, early_stop=early_stop)


if __name__ == "__main__":
    test_gemini_plugin(early_stop=False)
