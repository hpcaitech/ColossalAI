"""Native Gemma 4 dense policy for data parallel training (TP=1).

Keeping the native forward preserves per-layer embeddings and cross-layer KV autograd.
No fused kernels, tensor/sequence parallelism or pipeline parallelism are implemented here.
"""

from .base_policy import Policy


class Gemma4Policy(Policy):
    def config_sanity_check(self):
        config = self.shard_config
        if config.tensor_parallel_size != 1 or config.enable_sequence_parallelism or self.pipeline_stage_manager:
            raise NotImplementedError("Gemma 4 currently supports data parallelism with TP=1 and no pipeline only")
        if (
            any(
                getattr(config, name)
                for name in (
                    "enable_flash_attention",
                    "enable_fused_normalization",
                    "enable_jit_fused",
                    "fp8_communication",
                )
            )
            or config.gradient_checkpoint_config is not None
        ):
            raise NotImplementedError("Use native Gemma 4 kernels and HF non-reentrant gradient checkpointing")

    def preprocess(self):
        text_config = getattr(self.model.config, "text_config", self.model.config)
        if text_config.enable_moe_block:
            raise NotImplementedError("This policy covers Gemma 4 dense models only")
        return self.model

    def module_policy(self):
        return {}

    def postprocess(self):
        return self.model


class Gemma4ForCausalLMPolicy(Gemma4Policy):
    pass


class Gemma4ForConditionalGenerationPolicy(Gemma4Policy):
    pass
