import pytest
import torch
import transformers
from topo_utils import MLP, check_topo, split_model_and_get_DAG

BATCH_SIZE = 1
SEQ_LENGHT = 16


@pytest.mark.skip("ShapeProp is not compatible with PyTorch 1.11.0")
def test_opt():
    MODEL_LIST = [
        MLP,
        transformers.OPTModel,
    ]

    CONFIGS = [
        {"dim": 10, "layers": 12},
        transformers.OPTConfig(vocab_size=100, hidden_size=128, num_hidden_layers=4, num_attention_heads=4),
    ]

    def data_gen_MLP():
        x = torch.zeros((16, 10))
        kwargs = dict(x=x)
        return kwargs

    def data_gen_OPT():
        input_ids = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        attention_mask = torch.zeros((BATCH_SIZE, SEQ_LENGHT), dtype=torch.int64)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        return kwargs

    DATAGEN = [
        data_gen_MLP,
        data_gen_OPT,
    ]

    for i, model_cls in enumerate(MODEL_LIST):
        model = model_cls(config=CONFIGS[i])
        top_mod, topo = split_model_and_get_DAG(model, DATAGEN[i])
        # print(f'{top_mod=}\n----\n{topo=}')
        check_topo(top_mod, topo)


if __name__ == "__main__":
    test_opt()
