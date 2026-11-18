from ._remote import NativeRemotePolicy


class OuroForCausalLMPolicy(NativeRemotePolicy):
    """Keep the adapted Ouro loop and shared-layer parameters intact."""

    family = "ouro"

    def config_sanity_check(self):
        super().config_sanity_check()
        from ..modeling.ouro import _check_config

        if type(self.model).__module__ != "colossalai.shardformer.modeling.ouro":
            raise NotImplementedError("Construct Ouro with get_ouro_model_class before applying its ShardFormer policy")
        _check_config(self.model.config)
