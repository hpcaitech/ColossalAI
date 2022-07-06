import transformers
import torch
from colossalai.fx import ColoTracer
from torch.fx import GraphModule

BATCH_SIZE = 2
SEQ_LENGHT = 16


def trace_bert_and_compare_output(model, data_gen):
    tracer = ColoTracer()
    # make sure that the model is traceable
    try:
        kwargs = data_gen()
        meta_args = {k: v.to('meta') for k, v in kwargs.items()}
        graph = tracer.trace(root=model, meta_args=meta_args)
    except Exception as e:
        raise RuntimeError(f"Failed to trace {model.__class__.__name__}, error: {e}")
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # check output
    inputs = data_gen()

    # must turn on eval mode to ensure the output is consistent
    gm.eval()
    model.eval()

    # run forward
    non_fx_out = model(**inputs)
    fx_out = gm(**inputs)

    for k in non_fx_out.keys():
        assert torch.equal(fx_out[k], non_fx_out[k]), f'{model.__class__.__name__} has incorrect output {k}'


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
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return meta_args

    for model_cls in MODEL_LIST:
        model = model_cls(config=config)
        trace_bert_and_compare_output(model, data_gen)


def test_multi_sentence_bert():
    config = transformers.BertConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    def data_gen_for_next_sentence():
        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        return encoding

    model = transformers.BertForNextSentencePrediction(config)
    trace_bert_and_compare_output(model, data_gen_for_next_sentence)

    def data_gen_for_qa():
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = tokenizer(question, text, return_tensors="pt")
        return inputs

    model = transformers.BertForQuestionAnswering(config)
    trace_bert_and_compare_output(model, data_gen_for_qa)

    def data_gen_for_mcq():
        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        choice0 = "It is eaten with a fork and a knife."
        choice1 = "It is eaten while held in the hand."
        encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
        encoding = {k: v.unsqueeze(0) for k, v in encoding.items()}
        return encoding

    model = transformers.BertForMultipleChoice(config)
    trace_bert_and_compare_output(model, data_gen_for_mcq)


if __name__ == '__main__':
    test_single_sentence_bert()
    test_multi_sentence_bert()
