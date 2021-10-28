#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os.path as osp

import pytest
import torch
from torch.utils.data import DataLoader

import colossalai
from colossalai.builder import build_dataset, build_loss, build_data_sampler, build_model
from colossalai.core import global_context
from colossalai.engine.gradient_handler import DataParallelGradientHandler
from colossalai.nn.optimizer import ZeroRedundancyOptimizer_Level_1, ZeroRedundancyOptimizer_Level_3, \
    ZeroRedundancyOptimizer_Level_2
from colossalai.utils import print_rank_0

DIR_PATH = osp.dirname(osp.abspath(__file__))
CONFIG_PATH = osp.join(DIR_PATH, 'config.py')


def run_dist():
    colossalai.init_dist(CONFIG_PATH)

    # build resnet model
    model = build_model(global_context.config.model)
    model.build_from_cfg()
    model = model.cuda()

    level = global_context.config.level

    if level > 1:
        model = model.half()

    # test init cuda memory
    _ = torch.rand(1).cuda()
    torch.cuda.synchronize()
    max_alloc = torch.cuda.max_memory_allocated()
    max_reserved = torch.cuda.max_memory_reserved()
    print(f'before run: max_allocation = {max_alloc}, max_reserved = {max_reserved}')

    # build dataloader
    train_dataset = build_dataset(global_context.config.train_data.dataset)

    sampler_cfg = global_context.config.train_data.dataloader.pop('sampler', None)
    if sampler_cfg is None:
        train_dataloader = DataLoader(dataset=train_dataset, **global_context.config.train_data.dataloader)
    else:
        sampler = build_data_sampler(sampler_cfg, train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, sampler=sampler,
                                      **global_context.config.train_data.dataloader)

    test_dataset = build_dataset(global_context.config.test_data.dataset)
    test_dataloader = DataLoader(dataset=test_dataset, **global_context.config.test_data.dataloader)

    # build optimizer and loss
    # optimizer = build_optimizer(global_context.config.optimizer, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if level == 1:
        zero_optim = ZeroRedundancyOptimizer_Level_1(init_optimizer=optimizer, verbose=False)
    elif level == 2:
        zero_optim = ZeroRedundancyOptimizer_Level_2(init_optimizer=optimizer, cpu_offload=True, verbose=False)
    elif level == 3:
        zero_optim = ZeroRedundancyOptimizer_Level_3(init_optimizer=optimizer,
                                                     module=model,
                                                     verbose=False,
                                                     offload_optimizer_config=dict(
                                                         device='cpu',
                                                         pin_memory=True,
                                                         buffer_count=5,
                                                         fast_init=False
                                                     ),
                                                     offload_param_config=dict(
                                                         device='cpu',
                                                         pin_memory=True,
                                                         buffer_count=5,
                                                         buffer_size=1e8,
                                                         max_in_cpu=1e9
                                                     )
                                                     )

    loss_fn = build_loss(global_context.config.loss)
    gradient_handler = DataParallelGradientHandler(model, zero_optim)

    # train
    for epoch in range(100):
        model.train()

        # train
        avg_train_loss = 0
        train_iter = 0

        for idx, (data, label) in enumerate(train_dataloader):
            # model = model.half()
            data = data[0].cuda()
            label = label[0].cuda()

            if level > 1:
                data = data.half()

            output = model(data)
            loss = loss_fn(output[0], label)

            if level > 1:
                zero_optim.backward(loss)
                zero_optim.overlapping_partition_gradients_reduce_epilogue()
            else:
                loss.backward()
                gradient_handler.handle_gradient()

            zero_optim.step()
            zero_optim.zero_grad()

            avg_train_loss += loss.detach().cpu().numpy()
            train_iter += 1

        print_rank_0(f'epoch: {epoch}, train loss: {avg_train_loss / train_iter}')

        if epoch % 2 == 0:
            model.eval()
            avg_eval_loss = 0
            correct = 0
            total = 0
            eval_iters = 0

            for idx, (data, label) in enumerate(test_dataloader):
                with torch.no_grad():
                    data = data[0].cuda()
                    label = label[0].cuda()

                    if level > 1:
                        data = data.half()

                    output = model(data)
                    loss = loss_fn(output[0], label)

                avg_eval_loss += loss.detach().cpu().numpy()
                preds = torch.argmax(output[0], dim=1)
                total += data.size(0)
                correct += sum(preds == label)
                eval_iters += 1

            print_rank_0(f'epoch: {epoch}, eval loss: {avg_eval_loss / eval_iters}, acc: {correct / total}')


@pytest.mark.skip("This test should be invoked manually using the script provided")
@pytest.mark.dist
def test_zero():
    run_dist()


if __name__ == '__main__':
    test_zero()
