import pytest
import torch
import transformers
from hf_utils import split_model_and_compare_output

BATCH_SIZE = 64
SEQ_LENGHT = 16
NUM_EPOCHS = 2
NUM_CHUNKS = 1


@pytest.mark.skip("balance split v2 is not ready")
def test_gpt():
    MODEL_LIST = [
        transformers.GPT2Model,
        transformers.GPT2LMHeadModel,
        transformers.GPT2DoubleHeadsModel,
        transformers.GPT2ForTokenClassification,
        # transformers.GPT2ForSequenceClassification, # not supported yet
    ]
    config = transformers.GPT2Config(n_position=64, n_layer=4, n_head=8)

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return kwargs

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)
        split_model_and_compare_output(model, data_gen)


if __name__ == "__main__":
    test_gpt()
