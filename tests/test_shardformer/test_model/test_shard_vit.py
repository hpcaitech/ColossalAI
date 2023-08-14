import pytest
import torch

import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer.layer.utils import Randomizer
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

        if org_model.__class__.__name__ == 'ViTModel':
            check_output_hidden_state(org_output, sharded_output, stage_manager, atol=atol, rtol=rtol)

        check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

    # unwrap model
    if org_model.__class__.__name__ == 'ViTModel':
        vit_model = org_model
        shard_vit_model = sharded_model.unwrap()
    else:
        vit_model = org_model.vit
        shard_vit_model = sharded_model.unwrap().vit

    # check grad
    row_layer_for_check = ['encoder.layer[0].attention.attention.query', 'embeddings.patch_embeddings.projection']
    col_layer_for_check = ['encoder.layer[0].attention.output.dense']
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config['precision'] == 'fp32':
            atol, rtol = 1e-5, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_grad(vit_model,
                   shard_vit_model,
                   row_layer_for_check,
                   tp_group,
                   atol=atol,
                   rtol=rtol,
                   dim=0,
                   verbose=False)
        check_grad(vit_model,
                   shard_vit_model,
                   col_layer_for_check,
                   tp_group,
                   atol=atol,
                   rtol=rtol,
                   dim=1,
                   verbose=False)

    # check weights after optimizer.step()
    org_optimizer.step()
    sharded_optimizer.step()
    if stage_manager is None or stage_manager.is_first_stage():
        if test_config['precision'] == 'fp32':
            atol, rtol = 5e-3, 1e-3
        else:
            atol, rtol = 5e-3, 5e-3
        check_weight(vit_model,
                     shard_vit_model,
                     col_layer_for_check,
                     tp_group,
                     atol=atol,
                     rtol=rtol,
                     dim=1,
                     verbose=False)

    torch.cuda.empty_cache()


@parameterize('test_config', [{
    'tp_size': 2,
    'pp_size': 2,
    'num_microbatches': 4,
    'enable_all_optimization': True,
    'use_lazy_init': False,
    'precision': 'fp16',
    'initial_scale': 1,
}, {
    'tp_size': 1,
    'pp_size': 2,
    'num_microbatches': 4,
    'enable_all_optimization': False,
    'use_lazy_init': False,
    'precision': 'fp32',
}, {
    'tp_size': 4,
    'pp_size': 1,
    'enable_all_optimization': True,
    'use_lazy_init': False,
    'precision': 'fp32',
}])
def run_vit_test(test_config):

    # TODO: add test_config for TP+DP after supporting & debugging it
    # TODO: fix bug when settign lazy_init for Conv2D Layers in ViT models

    sub_model_zoo = model_zoo.get_sub_registry('transformers_vit')

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        check_forward_backward(model_fn, data_gen_fn, output_transform_fn, loss_fn, test_config)

    clear_layout_converter()
    Randomizer.reset_index()
    torch.cuda.empty_cache()


def check_vit(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_vit_test()


@pytest.mark.dist
@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_vit():
    spawn(check_vit, 4)


if __name__ == "__main__":
    test_vit()
