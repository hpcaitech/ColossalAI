import os

from transformers import BertForPreTraining, LlamaForCausalLM

import colossalai.interface.pretrained as pretrained_utils
from colossalai.lazy import LazyInitContext


def test_lazy_from_pretrained():
    # test from cached file, unsharded
    model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")
    with LazyInitContext():
        deffered_model = BertForPreTraining.from_pretrained("prajjwal1/bert-tiny")
    pretrained_path = pretrained_utils.get_pretrained_path(deffered_model)
    assert os.path.isfile(pretrained_path)
    for p, lazy_p in zip(model.parameters(), deffered_model.parameters()):
        assert p.shape == lazy_p.shape

    # test from local file, sharded
    llama_path = os.environ["LLAMA_PATH"]
    model = LlamaForCausalLM.from_pretrained(llama_path)
    with LazyInitContext():
        deffered_model = LlamaForCausalLM.from_pretrained(llama_path)
    pretrained_path = pretrained_utils.get_pretrained_path(deffered_model)
    assert os.path.isfile(pretrained_path)
    for p, lazy_p in zip(model.parameters(), deffered_model.parameters()):
        assert p.shape == lazy_p.shape


if __name__ == "__main__":
    test_lazy_from_pretrained()
