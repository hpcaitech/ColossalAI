import os
from typing import Optional

import torch
from peft import PeftModel

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.moe import MOE_MANAGER
from colossalai.moe.utils import skip_init
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossalai.shardformer import ShardConfig

from .huggingface import HuggingFaceModel

IGNORE_INDEX = -100


class MixtralModel(HuggingFaceModel):
    """
    Model wrapper around HuggingFace AutoModelForCausalLM models.

    Args:
        path: The path to a HuggingFace model.
        model_max_length: The maximum sequence length of the model.
        tokenizer_path: The path to the tokenizer.
        tokenizer_kwargs: Keyword arguments for the tokenizer.
        peft_path: The name or path to the HuggingFace's PEFT model.
        model_kwargs: Keyword arguments for the model.
        prompt_template: The model's prompt template.
        batch_size: Batch size for inference.
        logger: Logger for the model.
        shard_config: Shard config for tensor parallel.

    """

    def _load_model(
        self,
        path: str,
        model_kwargs: dict,
        peft_path: Optional[str] = None,
        shard_config: ShardConfig = None,
        moe_config: dict = None,
    ):
        """
        Load model.

        Args:
            path: The path to the model.
            model_kwargs: Keyword arguments for the model.
            peft_path: The path to the peft model.
            shard_config: Shard config for tensor parallel.

        """
        if "torch_dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = eval(model_kwargs["torch_dtype"])
        else:
            model_kwargs.setdefault("torch_dtype", torch.float16)

        if "config" in model_kwargs:
            model_kwargs["config"] = AutoConfig.from_pretrained(model_kwargs["config"])

        if moe_config is not None:
            coordinator = DistCoordinator()

            ep_size = moe_config["ep_size"]
            del moe_config["ep_size"]

            plugin = MoeHybridParallelPlugin(
                pp_size=1,
                **moe_config,
            )
            MOE_MANAGER.setup(
                parallel="EP",
                max_ep_size=ep_size,
                **{},
            )
            config = MixtralConfig.from_pretrained(path, **model_kwargs)
            config.num_local_experts = 1
            with skip_init():
                model = MixtralForCausalLM(config)

            model.to(torch.cuda.current_device())

            with skip_init():
                replace_moe_layer(model)

            # Set booster
            booster = Booster(plugin=plugin, **{})
            model, _, _, _, _ = booster.boost(model=model)

            if os.path.exists(os.path.join(path, "model.safetensors.index.json")):
                booster.load_model(model, os.path.join(path, "model.safetensors.index.json"))
            elif os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                booster.load_model(model, os.path.join(path, "pytorch_model.bin.index.json"))
            self.model = model.module
            coordinator.print_on_master("Finished loading model checkpoint")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).to(torch.cuda.current_device())
            if peft_path is not None:
                self.model = PeftModel.from_pretrained(self.model, peft_path, is_trainable=False)

        self.model.eval()