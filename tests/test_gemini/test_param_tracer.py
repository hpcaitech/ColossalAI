import numpy as np
import torch

from colossalai.gemini.memory_tracer.param_tracer_wrapper import ParamTracerWrapper
from colossalai.gemini.memory_tracer.model_data_memtracer import GLOBAL_CUDA_MEM_INFO
from colossalai.utils.model.colo_init_context import ColoInitContext
from tests.components_to_test.registry import non_distributed_component_funcs

def run_fwd_bwd(model, data, label, criterion, enable_autocast=False, dtype=torch.half):
    with torch.cuda.amp.autocast(enabled=enable_autocast):
        if criterion:
            y = model(data)
            loss = criterion(y, label)
        else:
            loss = model(data, label)
        loss = loss.to(dtype)
    model.backward(loss)

def run_param_wrapper_testing():
    test_models = ['simple_net', 'repeated_computed_layers', 'nested_model']

    for model_name in test_models:
        get_components_func = non_distributed_component_funcs.get_callable(model_name)
        model_builder, train_dataloader, _, _, criterion = get_components_func()

        with ColoInitContext(device=torch.device('cpu')):
            model = model_builder(checkpoint=False)

        model = ParamTracerWrapper(model)

        for i, (data, label) in enumerate(train_dataloader):
            if i > 1:
                break
            data = data.cuda()
            label = label.cuda()

            run_fwd_bwd(model, data, label, criterion, False)

        cuda_non_model_data_list = np.array(GLOBAL_CUDA_MEM_INFO.non_model_data_list) / 1024 ** 2
        print("cuda_non_model_data_list", len(cuda_non_model_data_list))
        # print(GLOBAL_CUDA_MEM_INFO.non_model_data_list)

        del model



if __name__ == '__main__':
    run_param_wrapper_testing()