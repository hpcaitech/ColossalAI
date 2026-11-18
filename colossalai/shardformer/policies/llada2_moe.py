from ._remote import NativeRemotePolicy


class LLaDA2MoeModelLMPolicy(NativeRemotePolicy):
    """Preserve the official LLaDA2.1/2.2 MoE and same-position label semantics."""

    family = "llada2_moe"

    def config_sanity_check(self):
        super().config_sanity_check()
        if type(self.model).__module__ != "colossalai.shardformer.modeling.llada2_moe":
            raise NotImplementedError(
                "Construct the model with get_llada2_model_class before applying its ShardFormer policy"
            )
        if self.model.config._attn_implementation != "eager":
            raise NotImplementedError("LLaDA2 compatibility requires attn_implementation='eager'")
