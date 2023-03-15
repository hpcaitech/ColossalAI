import pytest
from hf_tracer_utils import trace_model_and_compare_output

from tests.kit.model_zoo import model_zoo


# TODO: remove this skip once we handle the latest gpt model
@pytest.mark.skip
def test_gpt():
    sub_registry = model_zoo.get_sub_registry('transformers_gpt')

    for name, (model_fn, data_gen_fn, _, _) in sub_registry.items():
        model = model_fn()
        trace_model_and_compare_output(model, data_gen_fn)


if __name__ == '__main__':
    test_gpt()
