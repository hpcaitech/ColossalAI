import pytest
import torch
import transformers
from hf_tracer_utils import trace_model_and_compare_output

BATCH_SIZE = 2
SEQ_LENGTH = 16


def test_single_sentence_bert():
    MODEL_LIST = [
        transformers.BertModel,
        transformers.BertForPreTraining,
        transformers.BertLMHeadModel,
        transformers.BertForMaskedLM,
        transformers.BertForSequenceClassification,
        transformers.BertForTokenClassification,
    ]

    config = transformers.BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)

    def data_gen():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGTH), dtype=torch.int64)
        meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return meta_args

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)
        trace_model_and_compare_output(model, data_gen)


def test_multi_sentence_bert():
    config = transformers.BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    def data_gen_for_next_sentence():
        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        return encoding

    model = transformers.BertForNextSentencePrediction(config)
    trace_model_and_compare_output(model, data_gen_for_next_sentence)

    def data_gen_for_qa():
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = tokenizer(question, text, return_tensors="pt")
        return inputs

    model = transformers.BertForQuestionAnswering(config)
    trace_model_and_compare_output(model, data_gen_for_qa)

    def data_gen_for_mcq():
        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        choice0 = "It is eaten with a fork and a knife."
        choice1 = "It is eaten while held in the hand."
        encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
        encoding = {k: v.unsqueeze(0) for k, v in encoding.items()}
        return encoding

    model = transformers.BertForMultipleChoice(config)
    trace_model_and_compare_output(model, data_gen_for_mcq)


if __name__ == '__main__':
    test_single_sentence_bert()
    test_multi_sentence_bert()
