import os
import tempfile

from transformers import BertConfig, BertForPreTraining, LlamaConfig, LlamaForCausalLM

import colossalai.interface.pretrained as pretrained_utils
from colossalai.lazy import LazyInitContext


def test_lazy_from_pretrained():
    with tempfile.TemporaryDirectory() as tempdir:
        # Test an unsharded local checkpoint without relying on Hub state.
        bert_path = os.path.join(tempdir, "bert")
        BertForPreTraining(
            BertConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4)
        ).save_pretrained(bert_path)
        model = BertForPreTraining.from_pretrained(bert_path)
        with LazyInitContext():
            deferred_model = BertForPreTraining.from_pretrained(bert_path)
        pretrained_path = pretrained_utils.get_pretrained_path(deferred_model)
        assert os.path.isfile(pretrained_path)
        for p, lazy_p in zip(model.parameters(), deferred_model.parameters()):
            assert p.shape == lazy_p.shape

        # Test a sharded local checkpoint as well.
        llama_path = os.path.join(tempdir, "llama")
        LlamaForCausalLM(
            LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                vocab_size=128,
            )
        ).save_pretrained(llama_path, max_shard_size="20KB")
        model = LlamaForCausalLM.from_pretrained(llama_path)
        with LazyInitContext():
            deferred_model = LlamaForCausalLM.from_pretrained(llama_path)
        pretrained_path = pretrained_utils.get_pretrained_path(deferred_model)
        assert os.path.isfile(pretrained_path)
        for p, lazy_p in zip(model.parameters(), deferred_model.parameters()):
            assert p.shape == lazy_p.shape


if __name__ == "__main__":
    test_lazy_from_pretrained()
