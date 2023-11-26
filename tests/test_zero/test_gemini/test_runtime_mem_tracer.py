from copy import deepcopy

import numpy as np
import pytest
import torch

from colossalai.testing import DummyDataloader, clear_cache_before_run
from colossalai.zero.gemini.memory_tracer.runtime_mem_tracer import RuntimeMemTracer
from tests.kit.model_zoo import model_zoo, run_fwd_bwd


@pytest.mark.skip("this is not used")
@clear_cache_before_run()
def test_runtime_mem_tracer():
    test_models = ["gpt2", "bert", "simple_net", "repeated_computed_layers", "nested_model", "albert"]

    for model_name in test_models:
        model_builder, data_gen_fn, output_transform_fn, *_ = next(
            iter(model_zoo.get_sub_registry(model_name).values())
        )

        model = model_builder().cuda()

        model_bk = deepcopy(model)
        runtime_mem_tracer = RuntimeMemTracer(model)

        train_dataloader = DummyDataloader(data_gen_fn)
        for i, data in enumerate(train_dataloader):
            if i > 1:
                break
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            run_fwd_bwd(runtime_mem_tracer, data, output_transform_fn, optimizer=runtime_mem_tracer)

        for p1, p2 in zip(model_bk.parameters(), model.parameters()):
            torch.allclose(p1.to(torch.half), p2)

        non_model_data_list = runtime_mem_tracer._memstats.non_model_data_list("cuda")
        cuda_non_model_data_list = np.array(non_model_data_list) / 1024**2
        print("cuda_non_model_data_list", len(cuda_non_model_data_list))
        print(non_model_data_list)

        cnt1 = 0
        for p in runtime_mem_tracer.parameters_in_runtime_order():
            cnt1 += 1
        cnt2 = 0
        for p in model.parameters():
            cnt2 += 1
        assert cnt2 == cnt1, f"visited param number {cnt1} vs real param number {cnt2}"
        del model


if __name__ == "__main__":
    test_runtime_mem_tracer()
