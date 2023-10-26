import copy

import torch
from torch import distributed as dist
from torch.optim import AdamW

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.testing import assert_equal, assert_not_equal, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo


def check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type):
    model = model_fn()

    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)
    model = booster.enable_lora(model, task_type=task_type, r=8, lora_alpha=32, lora_dropout=0.1)
    model_copy = copy.deepcopy(model)

    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = loss_fn

    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    data = data_gen_fn()
    data = {k: v.to("cuda") if torch.is_tensor(v) or "Tensor" in v.__class__.__name__ else v for k, v in data.items()}

    output = model(**data)
    output = output_transform_fn(output)
    loss = criterion(output)

    booster.backward(loss, optimizer)
    optimizer.clip_grad_by_norm(1.0)
    optimizer.step()

    if dist.get_rank() == 0:
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_copy.named_parameters()):
            if "lora_" in n1:
                # lora modules require gradients, thus updated
                assert p1.requires_grad
                assert_not_equal(p1.to(p2.device), p2)
            else:
                if not p1.requires_grad:
                    assert_equal(p1.to(p2.device), p2)

    # # test saving and loading
    # with shared_tempdir() as tempdir:
    #     booster.save_lora(model, f"{tempdir}/model")
    torch.cuda.empty_cache()


def run_lora_test():
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")
    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        task_type = None
        if name == "transformers_llama_for_casual_lm":
            task_type = "CAUSAL_LM"
        if name == "transformers_llama_for_sequence_classification":
            task_type = "SEQ_CLS"
        check_fwd_bwd(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type)
    # check_checkpoint(model_fn, data_gen_fn, output_transform_fn, loss_fn, task_type)


def run_dist(rank, world_size, port):
    config = {}
    colossalai.launch(config=config, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_lora_test()


@rerun_if_address_is_in_use()
def test_torch_ddp_lora():
    spawn(run_dist, 2)
