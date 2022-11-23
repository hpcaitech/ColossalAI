import pytest
import torch
import transformers
from hf_tracer_utils import trace_model_and_compare_output

BATCH_SIZE = 1
SEQ_LENGTH = 16


def test_opt():
    MODEL_LIST = [
        transformers.OPTModel,
        transformers.OPTForCausalLM,
    ]

    config = transformers.OPTConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4)

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)
        trace_model_and_compare_output(model, data_gen)


if __name__ == '__main__':
    test_opt()
