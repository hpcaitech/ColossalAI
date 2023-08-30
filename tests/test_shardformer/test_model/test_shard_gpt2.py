import pytest
import torch
from torch import distributed as dist

import colossalai
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.logging import disable_existing_loggers
from colossalai.tensor.d_tensor.api import clear_layout_converter
from colossalai.testing import clear_cache_before_run, parameterize, rerun_if_address_is_in_use, spawn
from tests.kit.model_zoo import model_zoo
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
    check_grad,
    check_loss,
    check_output_hidden_state,
    check_weight,
    run_forward_backward_with_hybrid_plugin,
)


def check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config):

    org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = \
        build_model_from_hybrid_plugin(model_fn, loss_fn, test_config)

    org_loss, org_output, sharded_loss, sharded_output = \
        run_forward_backward_with_hybrid_plugin(
            org_model,
            sharded_model,
            sharded_optimizer,
            data_gen_fn,
            output_transform_fn,
            criterion,
            booster)

    stage_manager = booster.plugin.stage_manager
    tp_group = booster.plugin.tp_group

    # check last hidden state & loss
    if stage_manager is None or stage_manager.is_last_stage():
        if test_config['precision'] == 'fp32':
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3

        if org_model.__class__.__name__ == 'GPT2Model':
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    def unwrap(module):
        if isinstance(module, HybridParallelModule):
            module = module.unwrap()
        if module.__class__.__name__ == 'GPT2Model':
            return module
        return module.transformer

    # unwrap model
    gpt2 = unwrap(org_model)
    sharded_gpt2 = unwrap(sharded_model)

    col_layer_for_check = ['h[0].mlp.c_fc']
    row_layer_for_check = ['wte', 'h[0].mlp.c_proj']

    # check grad
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config['precision'] == 'fp32':
            atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_grad(gpt2, sharded_gpt2, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False)
        check_grad(gpt2, sharded_gpt2, row_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=0, verbose=False)

    # check weights after optimizer.step()
    org_optimizer.step()
    sharded_optimizer.step()
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config['precision'] == 'fp32':
            atol, rtol = 5e-3, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_weight(gpt2, sharded_gpt2, col_layer_for_check, tp_group, atol=atol, rtol=rtol, dim=1, verbose=False)

    torch.cuda.empty_cache()


@parameterize('test_config', [{
    'tp_size': 2,
    'pp_size': 2,
    'num_microbatches': 4,
    'enable_all_optimization': True,
    'use_lazy_init': True,
    'precision': 'fp16',
    'initial_scale': 1,
}, {
    'tp_size': 1,
    'pp_size': 2,
    'num_microbatches': 4,
    'enable_all_optimization': True,
    'use_lazy_init': True,
    'precision': 'fp16',
    'initial_scale': 1,
}, {
    'tp_size': 4,
    'pp_size': 1,
    'enable_all_optimization': True,
    'use_lazy_init': False,
    'precision': 'fp32',
}])
@clear_cache_before_run()
def run_gpt2_test(test_config):

    # TODO(baizhou): add test_config for TP+DP after supporting & debugging it

    sub_model_zoo = model_zoo.get_sub_registry('transformers_gpt')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    torch.cuda.empty_cache()


def check_gpt2(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_gpt2_test()


# TODO(ver217): fix this


@pytest.mark.skip("this will stuck in CI")
@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_gpt2():
    spawn(check_gpt2, 4)


if __name__ == "__main__":
    test_gpt2()
