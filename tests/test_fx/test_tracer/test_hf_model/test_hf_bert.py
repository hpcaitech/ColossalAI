import transformers
import torch
from colossalai.fx import ColoTracer
from torch.fx import GraphModule

BATCH_SIZE = 2
SEQ_LENGHT = 16


def test_bert():
    tracer = ColoTracer()
    config = transformers.BertConfig()
    model = transformers.BertModel(config=config)

    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64, device='meta')
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64, device='meta')
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64, device='meta')
    meta_args = dict(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    # make sure that the model is traceable
    graph = tracer.trace(root=model, meta_args=meta_args)
    gm = GraphModule(model, graph, model.__class__.__name__)
    gm.recompile()

    # check output
    input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)

    # must turn on eval mode to ensure the output is consistent
    gm.eval()
    model.eval()

    # run forward
    fx_out = gm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    non_fx_out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    assert fx_out['last_hidden_state'].shape == non_fx_out['last_hidden_state'].shape
    assert torch.equal(fx_out['last_hidden_state'], non_fx_out['last_hidden_state'])


if __name__ == '__main__':
    test_bert()
