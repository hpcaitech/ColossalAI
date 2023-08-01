import copy
from contextlib import nullcontext

import pytest
import torch
from torch import distributed as dist
from torch.optim import Adam

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.lazy.lazy_init import LazyInitContext
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.api import (
    clear_layout_converter,
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import build_model, check_state_dict, run_forward


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):

    use_lazy_init = False
    if 'use_lazy_init' in test_config:
        use_lazy_init = test_config.pop('use_lazy_init')

    if use_lazy_init:
        ctx = LazyInitContext()
    else:
        ctx = nullcontext()

    # prepare booster
    plugin = HybridParallelPlugin(**test_config)
    booster = Booster(plugin=plugin)
    stage_manager = plugin.stage_manager

    # prepare models and optimizers
    with ctx:
        org_model = model_fn().cuda()
        sharded_model = copy.deepcopy(org_model)

    if use_lazy_init:
        org_model = ctx.materialize(org_model)

    org_optimizer = Adam(org_model.parameters(), lr=1e-3)
    sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)
    criterion = loss_fn

    sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(sharded_model, sharded_optimizer, criterion)

    def _criterion(outputs, inputs):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    # do forward and backward
    data = data_gen_fn()
    sharded_model.train()
    if stage_manager:
        data = {
            k: v.to('cuda').repeat(4, 1) if torch.is_tensor(v) or 'Tensor' in v.__class__.__name__ else v
            for k, v in data.items()
        }
        data_iter = iter([data])
        sharded_output = booster.execute_pipeline(data_iter,
                                                  sharded_model,
                                                  _criterion,
                                                  sharded_optimizer,
                                                  return_loss=True,
                                                  return_outputs=True)
        sharded_loss = sharded_output['loss']
    else:
        data = {k: v.cuda() for k, v in data.items()}
        sharded_output = sharded_model(**data)
        sharded_loss = criterion(sharded_output)
        sharded_loss.backward()

    org_model.train()
    org_output = org_model(**data)
    org_loss = criterion(org_output)
    org_loss.backward()

    if stage_manager is None or stage_manager.is_last_stage():

        # check last hidden state
        if org_model.__class__.__name__ == 'GPT2Model':
            org_hidden_state = org_output.last_hidden_state

            if stage_manager is None:
                sharded_hidden_state = sharded_output.last_hidden_state

            if stage_manager and stage_manager.is_last_stage():
                sharded_hidden_state = torch.cat([output.last_hidden_state for output in sharded_output['outputs']],
                                                 dim=0)

            assert torch.allclose(org_hidden_state, sharded_hidden_state, atol=1e-5, rtol=1e-3), \
                f"shard model's output hidden state is not equal to origin model's last hidden state\n{org_hidden_state}\n{sharded_hidden_state}"

        # check loss
        assert torch.allclose(org_loss, sharded_loss, atol=1e-5, rtol=1e-3), \
            f"shard model loss is not equal to origin model loss\n{org_loss}\n{sharded_loss}"

    # unwrap model
    if org_model.__class__.__name__ == 'GPT2Model':
        org_model = org_model
        sharded_model = sharded_model.unwrap()
    else:
        org_model = org_model.transformer
        sharded_model = sharded_model.unwrap().transformer

    # check weights and gradients
    if stage_manager is None or stage_manager.is_first_stage():

        shard_weight = sharded_model.h[0].mlp.c_fc.weight
        org_grad = org_model.h[0].mlp.c_fc.weight.grad
        shard_grad = sharded_model.h[0].mlp.c_fc.weight.grad

        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
            shard_grad_list = [torch.zeros([*shard_grad.shape]).to('cuda') for _ in range(plugin.tp_size)]
            dist.all_gather(shard_grad_list, shard_grad, plugin.tp_group)
            shard_grad = torch.cat(shard_grad_list, dim=1)

        assert torch.allclose(org_grad, shard_grad, atol=1e-5, rtol=1e-3), \
            f"shard model grad is not equal to origin model grad\n{org_grad}\n{shard_grad}"

    # check weights after optimizer.step()
    org_optimizer.step()
    sharded_optimizer.step()
    if stage_manager is None or stage_manager.is_first_stage():

        org_weight = org_model.h[0].mlp.c_fc.weight
        shard_weight = sharded_model.h[0].mlp.c_fc.weight

        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(shard_weight):
            shard_weight_list = [torch.zeros([*shard_weight.shape]).to('cuda') for _ in range(plugin.tp_size)]
            dist.all_gather(shard_weight_list, shard_weight, plugin.tp_group)
            shard_weight = torch.cat(shard_weight_list, dim=1)

        assert torch.allclose(org_weight, shard_weight, atol=5e-3, rtol=1e-3), \
            f"shard model weight is not equal to origin model weight\n{org_weight}\n{shard_weight}"

    torch.cuda.empty_cache()


@parameterize('test_config', [{
    'tp_size': 1,
    'pp_size': 2,
    'num_microbatches': 4,
    'use_lazy_init': True
}, {
    'tp_size': 2,
    'pp_size': 2,
    'num_microbatches': 4,
    'enable_fused_normalization': False,
    'use_lazy_init': False
}, {
    'tp_size': 4,
    'pp_size': 1,
    'enable_fused_normalization': True,
    'use_lazy_init': False
}])
@clear_cache_before_run()
def run_gpt2_test(test_config):

    # TODO: add plugin_config for TP+DP after supporting & debugging it
    # {'tp_size': 2, 'pp_size': 1, 'enable_fused_normalization': True}

    sub_model_zoo = model_zoo.get_sub_registry('transformers_gpt')
    test_config['precision'] = 'float'    # Do not use fp16/bf16 in testing

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_gpt2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt2_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2():
    spawn(check_gpt2, 4)


if __name__ == "__main__":
    test_gpt2()
