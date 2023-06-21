from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model

import colossalai.shardformer.layer as col_nn

from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription


class GPT2Policy(Policy):

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        return {
            GPT2Model:
                ModulePolicyDescription(attribute_replacement={},
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="wte",
                                                target_module=col_nn.VocabParallelEmbedding1D,
                                            ),
                                        ]),
            GPT2Block:
                ModulePolicyDescription(attribute_replacement={
                    "attn.embed_dim": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "attn.split_size": self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                    "attn.num_heads": self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                },
                                        param_replacement=[],
                                        sub_module_replacement=[
                                            SubModuleReplacementDescription(
                                                suffix="attn.c_attn",
                                                target_module=col_nn.LinearConv1D_Col,
                                                kwargs={
                                                    "n_cast": 3,
                                                },
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attn.c_proj",
                                                target_module=col_nn.LinearConv1D_Row,
                                                kwargs={
                                                    "n_cast": 1,
                                                },
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="mlp.c_fc",
                                                target_module=col_nn.LinearConv1D_Col,
                                                kwargs={
                                                    "n_cast": 1,
                                                },
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="mlp.c_proj",
                                                target_module=col_nn.LinearConv1D_Row,
                                                kwargs={
                                                    "n_cast": 1,
                                                },
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attn.attn_dropout",
                                                target_module=col_nn.Dropout1D,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="attn.resid_dropout",
                                                target_module=col_nn.Dropout1D,
                                            ),
                                            SubModuleReplacementDescription(
                                                suffix="mlp.dropout",
                                                target_module=col_nn.Dropout1D,
                                            ),
                                        ])
        }

    def new_model_class(self):

        return self.model

    def postprocess(self):
        return self.model


# GPT2Model
class GPT2ModelPolicy(GPT2Policy):

    def __init__(self) -> None:
        super().__init__()
