import pytest
import torch
from hf_tracer_utils import trace_model_and_compare_output
from packaging import version

from colossalai.testing import clear_cache_before_run
from tests.kit.model_zoo import model_zoo


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.12.0'), reason='torch version < 12')
@clear_cache_before_run()
def test_t5():
    sub_registry = model_zoo.get_sub_registry('transformers_t5')

    for name, (model_fn, data_gen_fn, _, _, _) in sub_registry.items():
        if name == "transformers_t5_for_conditional_generation":
            # cannot trace for loss function yet
            # so we use a data gen which does not produce labels
            data_gen_fn = sub_registry.get('transformers_t5')[1]

        model = model_fn()
        trace_model_and_compare_output(model, data_gen_fn, ignore_data=['labels'])


if __name__ == '__main__':
    test_t5()
