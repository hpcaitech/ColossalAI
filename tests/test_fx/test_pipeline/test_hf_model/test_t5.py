import pytest
import torch
import transformers
from hf_utils import split_model_and_compare_output

BATCH_SIZE = 1
SEQ_LENGHT = 16


@pytest.mark.skip("balance split v2 is not ready")
def test_t5():
    MODEL_LIST = [
        transformers.T5Model,
        transformers.T5ForConditionalGeneration,
        transformers.T5EncoderModel,
    ]

    config = transformers.T5Config(vocab_size=100, d_model=128, num_layers=2)

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        decoder_input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        return kwargs

    def data_gen_for_encoder_only():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids)
        return kwargs

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)

        if isinstance(model, transformers.T5EncoderModel):
            data_gen_func = data_gen_for_encoder_only
        else:
            data_gen_func = data_gen

        split_model_and_compare_output(model, data_gen_func)


if __name__ == "__main__":
    test_t5()
