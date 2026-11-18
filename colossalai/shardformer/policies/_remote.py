"""Conservative native policies for the phase-two remote-code model families."""

from .base_policy import Policy


class NativeRemotePolicy(Policy):
    """Admit the native single-device path; no kernel or parallel replacement."""

    family = None

    def config_sanity_check(self):
        from transformers import __version__

        if __version__ != "5.17.0":
            raise NotImplementedError(
                f"{self.family} policy is validated on transformers==5.17.0, installed {__version__}. "
                "See docs/HF5-COMPATIBILITY.md."
            )
        unsupported = [
            name
            for name in (
                "enable_tensor_parallelism",
                "enable_sequence_parallelism",
                "enable_flash_attention",
                "enable_jit_fused",
                "enable_fused_normalization",
                "enable_all_optimization",
                "fp8_communication",
            )
            if getattr(self.shard_config, name, False)
        ]
        unsupported.extend(
            name
            for name in ("pipeline_stage_manager", "ep_group", "moe_dp_group", "gradient_checkpoint_config")
            if getattr(self.shard_config, name, None) is not None
        )
        if self.shard_config.extra_kwargs:
            unsupported.append("extra_kwargs")
        if unsupported:
            raise NotImplementedError(
                f"{self.family} policy supports only the native nonparallel path; requested {', '.join(unsupported)}. "
                "See docs/HF5-COMPATIBILITY.md."
            )
        if getattr(self.model.config, "use_cache", False):
            raise NotImplementedError(f"{self.family} policy requires config.use_cache=False")
        if getattr(self.model, "is_gradient_checkpointing", False):
            raise NotImplementedError(f"{self.family} gradient checkpointing has not been validated")

    def preprocess(self):
        return self.model

    def module_policy(self):
        return {}

    def postprocess(self):
        return self.model
