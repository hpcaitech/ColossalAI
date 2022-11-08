import pytest
import torch
import transformers
from hf_tracer_utils import trace_model_and_compare_output

BATCH_SIZE = 1
SEQ_LENGTH = 16


def test_gpt():
    MODEL_LIST = [
        transformers.GPT2Model,
        transformers.GPT2LMHeadModel,
        transformers.GPT2DoubleHeadsModel,
        transformers.GPT2ForTokenClassification,
    # transformers.GPT2ForSequenceClassification, # not supported yet
    ]

    config = transformers.GPT2Config(n_position=64, n_layer=2, n_head=4)

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return kwargs

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)
        trace_model_and_compare_output(model, data_gen)


if __name__ == '__main__':
    test_gpt()
