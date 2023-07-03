import torch.nn as nn

import colossalai.shardformer.layer as col_nn

from .._utils import getattr_, setattr_
from .basepolicy import ModulePolicyDescription, Policy, SubModuleReplacementDescription

__all__ = [
    'BertPolicy', 'BertModelPolicy', 'BertForPretrainingPolicy', 'BertLMHeadModelPolicy', 'BertForMaskedLMPolicy',
    'BertForNextSentencePredictionPolicy', 'BertForSequenceClassificationPolicy', 'BertForTokenClassificationPolicy',
    'BertForMultipleChoicePolicy'
]


class BertPolicy(Policy):

    def config_sanity_check(self):
        pass

    def preprocess(self):
        # reshape the embedding layer
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        # TODO:
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
        return self.model

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer

        base_policy = {
            BertLayer:
                ModulePolicyDescription(
                    attribute_replacement={
        # 1. shard hidden size
                        "attention.self.all_head_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
                        "crossattention.self.all_head_size":
                            self.model.config.hidden_size // self.shard_config.tensor_parallel_size,
        # 2. shard number of heads
                        "attention.self.num_attention_heads":
                            self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                        "crossattention.self.num_attention_heads":
                            self.model.config.num_attention_heads // self.shard_config.tensor_parallel_size,
                    },
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="attention.self.query",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.self.key",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.self.value",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.self.dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.output.dense",
                            target_module=col_nn.Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="attention.output.dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        ),
                        SubModuleReplacementDescription(
                            suffix="intermediate.dense",
                            target_module=col_nn.Linear1D_Col,
                        ),
                        SubModuleReplacementDescription(
                            suffix="output.dense",
                            target_module=col_nn.Linear1D_Row,
                        ),
                        SubModuleReplacementDescription(
                            suffix="output.dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]),
            BertEmbeddings:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    )
                ])
        }

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            base_policy[BertLayer].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="attention.output.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ))
            base_policy[BertLayer].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="output.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ))
            base_policy[BertEmbeddings].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ),)
        return base_policy

    def postprocess(self):
        return self.model


# BertModel
class BertModelPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()


# BertForPreTraining
class BertForPretrainingPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        module_policy = super().module_policy()
        addon_module = {
            BertLMPredictionHead:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="decoder", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}),
                ])
        }

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            addon_module[BertLMPredictionHead].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="transform.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ))

        # append extra policy
        module_policy.update(addon_module)
        return module_policy

    def postprocess(self):
        binding_map = {"bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            param = nn.Parameter(param)
            setattr_(self.model, k, param)
            setattr_(self.model, v, param)
        return self.model


# BertLMHeadModel
class BertLMHeadModelPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        module_policy = super().module_policy()
        addon_module = {
            BertLMPredictionHead:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="decoder", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}),
                ])
        }
        if self.shard_config.enable_fused_normalization:
            addon_module[BertLMPredictionHead].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="transform.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ))
        module_policy.update(addon_module)
        return module_policy

    def postprocess(self):
        binding_map = {"bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            param = nn.Parameter(param)
            setattr_(self.model, k, param)
            setattr_(self.model, v, param)
        return self.model


# BertForMaskedLM
class BertForMaskedLMPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        module_policy = super().module_policy()
        addon_module = {
            BertLMPredictionHead:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="decoder", target_module=col_nn.Linear1D_Col, kwargs={"gather_output": True}),
                ])
        }

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            addon_module[BertLMPredictionHead].sub_module_replacement.append(
                SubModuleReplacementDescription(
                    suffix="transform.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ))

        module_policy.update(addon_module)
        return module_policy

    def postprocess(self):
        binding_map = {"bert.embeddings.word_embeddings.weight": "cls.predictions.decoder.weight"}
        for k, v in binding_map.items():
            param = getattr_(self.model, k)
            param = nn.Parameter(param)
            setattr_(self.model, k, param)
            setattr_(self.model, v, param)
        return self.model


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForSequenceClassification

        module_policy = super().module_policy()
        addon_module = {
            BertForSequenceClassification:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    )
                ])
        }
        module_policy.update(addon_module)
        return module_policy


# BertForTokenClassification
class BertForTokenClassificationPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForTokenClassification

        module_policy = super().module_policy()
        addon_module = {
            BertForTokenClassification:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    )
                ])
        }
        module_policy.update(addon_module)
        return module_policy


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()


# BertForMultipleChoice
class BertForMultipleChoicePolicy(BertPolicy):

    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForMultipleChoice

        module_policy = super().module_policy()
        addon_module = {
            BertForMultipleChoice:
                ModulePolicyDescription(sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    )
                ])
        }
        module_policy.update(addon_module)
        return module_policy
