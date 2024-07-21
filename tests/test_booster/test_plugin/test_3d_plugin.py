import copy
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
from torch.testing import assert_close
from torch.utils.data import Dataset

import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.fx import is_compatible_with_meta
from colossalai.lazy.lazy_init import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from colossalai.utils import set_seed
from tests.kit.model_zoo import model_zoo


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 100, max_length: int = 512, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.max_length = max_length
        set_seed(42)
        self.input_ids = torch.randint(
            0, vocab_size, (num_samples, max_length), device=get_accelerator().get_current_device()
        )
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def move_to_cuda(batch):
    return {k: v.cuda() for k, v in batch.items()}


@clear_cache_before_run()
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

        booster.execute_pipeline(data_iter, model, _criterion, optimizer, return_loss=True)
        optimizer.step()

    except Exception as e:
        return repr(e)


@parameterize("init_method", ["none", "lazy"])
def check_3d_plugin(init_method: str = "none", early_stop: bool = True):
    """check hybrid plugin over model zoo

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
        "transformers_llama_for_causal_lm"
    ).items():
        err = run_fn(init_method, model_fn, data_gen_fn, output_transform_fn)

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


@parameterize(
    "test_args",
    [
        {
            "batch_size": 8,
            "num_steps": 4,
            "tp": 2,
            "pp": 2,
            "pp_style": "1f1b",
            "num_model_chunks": 1,
            "num_microbatches": 4,
            "zero": 1,
            "precision": "fp16",
            "initial_scale": 1,
            "max_length": 512,
            "gradient_accumulation_step": 2,
        },
        {
            "batch_size": 8,
            "num_steps": 4,
            "tp": 2,
            "pp": 2,
            "pp_style": "1f1b",
            "num_model_chunks": 1,
            "num_microbatches": 4,
            "zero": 0,
            "precision": "fp16",
            "initial_scale": 1,
            "max_length": 512,
            "gradient_accumulation_step": 2,
        },
        {
            "batch_size": 8,
            "num_steps": 4,
            "tp": 1,
            "pp": 2,
            "pp_style": "1f1b",
            "num_model_chunks": 1,
            "num_microbatches": 4,
            "zero": 1,
            "precision": "fp16",
            "initial_scale": 1,
            "max_length": 512,
            "gradient_accumulation_step": 2,
        },
        {
            "batch_size": 1,
            "num_steps": 4,
            "tp": 2,
            "pp": 1,
            "pp_style": "1f1b",
            "num_model_chunks": 1,
            "num_microbatches": 1,
            "zero": 2,
            "precision": "fp16",
            "initial_scale": 1,
            "max_length": 512,
            "gradient_accumulation_step": 2,
        },
        {
            "batch_size": 1,
            "num_steps": 4,
            "tp": 2,
            "pp": 1,
            "pp_style": "1f1b",
            "num_model_chunks": 1,
            "num_microbatches": 1,
            "zero": 0,
            "precision": "fp16",
            "initial_scale": 1,
            "max_length": 512,
            "gradient_accumulation_step": 2,
        },
    ],
)
def run_grad_acc_test(test_args):
    model_fn, *_ = next(iter(model_zoo.get_sub_registry("transformers_gpt_lm").values()))
    model = model_fn()
    optimizer = HybridAdam(model.parameters())
    origin_model = copy.deepcopy(model).cuda()
    origin_optimizer = HybridAdam(origin_model.parameters())

    plugin = HybridParallelPlugin(
        tp_size=test_args["tp"],
        pp_size=test_args["pp"],
        pp_style=test_args["pp_style"],
        zero_stage=test_args["zero"],
        num_model_chunks=test_args["num_model_chunks"],
        enable_fused_normalization=True,
        num_microbatches=test_args["num_microbatches"],
        precision=test_args["precision"],
    )
    booster = Booster(plugin=plugin)

    dataset = RandomDataset(
        num_samples=test_args["batch_size"] * test_args["num_steps"] * plugin.dp_size,
        max_length=test_args["max_length"],
        vocab_size=model.config.vocab_size,
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=test_args["batch_size"], shuffle=True, drop_last=True)

    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

    grad_accu_step = test_args["gradient_accumulation_step"]
    for step, batch in enumerate(dataloader):
        batch = move_to_cuda(batch)
        # train origin model
        origin_output = origin_model(**batch)
        origin_loss = origin_output[0] / grad_accu_step
        origin_loss.backward()

        if (step + 1) % grad_accu_step != 0 and test_args["zero"] != 2:
            ctx = booster.no_sync(model, optimizer)
        else:
            ctx = nullcontext()

        with ctx:
            if plugin.stage_manager is not None:
                batch = iter([batch])
                booster.execute_pipeline(
                    batch,
                    model,
                    criterion=lambda outputs, inputs: outputs[0] / grad_accu_step,
                    optimizer=optimizer,
                    return_loss=False,
                )
            else:
                outputs = model(**batch)
                loss = outputs[0] / grad_accu_step
                booster.backward(loss, optimizer)

        if (step + 1) % grad_accu_step == 0:
            # update origin model weight
            origin_optimizer.step()
            origin_optimizer.zero_grad()

            # update sharded model
            optimizer.step()
            optimizer.zero_grad()

    # tricky code here, shard the origin model inorder to check the parameters in the same stage.
    origin_model, origin_optimizer, _, dataloader, _ = booster.boost(
        origin_model, origin_optimizer, dataloader=dataloader
    )
    for p1, p2 in zip(model.unwrap().parameters(), origin_model.unwrap().parameters()):
        assert_close(p1.to(p2.dtype), p2, atol=1e-2, rtol=1e-2)


def run_dist(rank, world_size, port, early_stop: bool = True):
    # init dist env
    colossalai.launch(rank=rank, world_size=world_size, port=port, host="localhost")
    check_3d_plugin(early_stop=early_stop)
    run_grad_acc_test()


@rerun_if_address_is_in_use()
def test_3d_plugin(early_stop: bool = True):
    spawn(run_dist, 4, early_stop=early_stop)


if __name__ == "__main__":
    test_3d_plugin(early_stop=False)
