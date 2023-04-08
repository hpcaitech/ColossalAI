from copy import deepcopy

import numpy as np
import torch

from colossalai.testing import clear_cache_before_run
from colossalai.zero import ColoInitContext
from colossalai.zero.gemini.memory_tracer.runtime_mem_tracer import RuntimeMemTracer
from tests.components_to_test import run_fwd_bwd
from tests.components_to_test.registry import non_distributed_component_funcs


@clear_cache_before_run()
def test_runtime_mem_tracer():
    test_models = ['gpt2', 'bert', 'simple_net', 'repeated_computed_layers', 'nested_model', 'albert']

    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        with ColoInitContext(device='cpu'):
            model = model_builder(checkpoint=False)

        model_bk = deepcopy(model)
        runtime_mem_tracer = RuntimeMemTracer(model)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 1:
                break
            data = data.cuda()
            label = label.cuda()

            run_fwd_bwd(runtime_mem_tracer, data, label, criterion, optimizer=runtime_mem_tracer)

        for p1, p2 in zip(model_bk.parameters(), model.parameters()):
            torch.allclose(p1.to(torch.half), p2)

        non_model_data_list = runtime_mem_tracer._memstats.non_model_data_list('cuda')
        cuda_non_model_data_list = np.array(non_model_data_list) / 1024**2
        print("cuda_non_model_data_list", len(cuda_non_model_data_list))
        print(non_model_data_list)

        cnt1 = 0
        for p in runtime_mem_tracer.parameters_in_runtime_order():
            cnt1 += 1
        cnt2 = 0
        for p in model.parameters():
            cnt2 += 1
        assert cnt2 == cnt1, f'visited param number {cnt1} vs real param number {cnt2}'
        del model


if __name__ == '__main__':
    test_runtime_mem_tracer()
