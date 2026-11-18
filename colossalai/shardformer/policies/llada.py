from ._remote import NativeRemotePolicy


class LLaDAModelLMPolicy(NativeRemotePolicy):
    """Preserve LLaDA's bidirectional attention and caller-defined diffusion loss."""

    family = "llada"

    def config_sanity_check(self):
        super().config_sanity_check()
        if type(self.model).__module__ != "colossalai.shardformer.modeling.llada":
            raise NotImplementedError(
                "Construct the model with get_llada_model_class before applying its ShardFormer policy"
            )
        if getattr(self.model.config, "flash_attention", False):
            raise NotImplementedError("LLaDA's native flash_attention path has not been validated")
