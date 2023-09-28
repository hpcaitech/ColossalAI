import pytest
import torch
from hf_tracer_utils import trace_model_and_compare_output
from packaging import version

from colossalai.testing import clear_cache_before_run
from tests.kit.model_zoo import model_zoo


@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("1.12.0"), reason="torch version < 12")
@clear_cache_before_run()
def test_opt():
    sub_registry = model_zoo.get_sub_registry("transformers_opt")
    for name, (model_fn, data_gen_fn, _, _, _) in sub_registry.items():
        model = model_fn()
        trace_model_and_compare_output(model, data_gen_fn, ignore_data=["labels", "start_positions", "end_positions"])


if __name__ == "__main__":
    test_opt()
