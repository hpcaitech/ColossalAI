
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM
from colossal_eval.utils import is_rank_0

from colossalai.shardformer import ShardConfig
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossalai.moe.utils import skip_init
from colossalai.moe import MOE_MANAGER
from colossalai.cluster import DistCoordinator
from colossalai.booster import Booster

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
        self, path: str, model_kwargs: dict, peft_path: Optional[str] = None, shard_config: ShardConfig = None, moe_config: dict = None
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
            self.model = model.module
            coordinator.print_on_master("Finished loading model checkpoint")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).to(torch.cuda.current_device())
            if peft_path is not None:
                self.model = PeftModel.from_pretrained(self.model, peft_path, is_trainable=False)

        self.model.eval()
    
    @torch.no_grad()
    def generate(self, inputs: List[str], max_new_tokens: int, **kwargs) -> List[str]:
        """Generate results given a list of inputs and get logits of the first new token over choices.

        Args:
            inputs: A list of strings.
            max_new_tokens: Max new tokens for generation.
            kwargs: Key arguments for generation

        Returns:
            A list of generated strings and logits over choices.

        Note:
            Currently the function only returns the logits of the first new token.
            It is used for single choice question.
            For multiple choices question, please avoid using the loss over choices.
            You should set argument choices as None in self.inference().

        """
        truncated_inputs = self._get_truncated_prompts(inputs, max_new_tokens)

        # Set to max length for EP
        encoded_inputs = self.tokenizer(
            truncated_inputs,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=self.model_max_length - max_new_tokens,
        ).to(torch.cuda.current_device())

        # Set output_scores=True to get prediction scores.
        outputs = self.model.generate(
            **encoded_inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            use_cache=True,
            **kwargs,
        )

        # We only need to decode predicted tokens.
        sequences = outputs.sequences[:, encoded_inputs["input_ids"].shape[1] :]

        scores = []
        if self.indices_for_choices:
            # If the question is a single-choice question, we will return the scores of specific indices for first predicted token.
            # The indices are the tokenization results of the options for the single-choice question.
            # For example, if the options of the question are A, B, C and D, we only returns scores at indices of A, B, C and D.
            for option_indices in self.indices_for_choices:
                scores.append(outputs.scores[0][:, option_indices].detach().cpu())

            scores = torch.max(torch.stack(scores), dim=0)[0]

        decoded_sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        return decoded_sequences, scores

    def _calculate_loss(self, input_ids_list: List[torch.LongTensor], labels: List[torch.LongTensor]) -> Tuple[List]:
        """
        Calculate loss only on target tokens.
        Hugging Face generate() function can't return per sample loss.
        It will only return the mean of the loss in a batch.
        In torch.nn.CrossEntropyLoss(), reduction should be specified as "none" to get per sample loss.

        Args:
            input_ids_list: A batch of input token ids.
            labels: A batch of labels.

        Returns:
            A list of loss.

        """

        # Left padding to the longest sequence in the batch
        reversed_input_ids = [seq.flip(dims=(0,)) for seq in input_ids_list]
        reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=reversed_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )  # (bsz, max_len)
        input_ids = torch.flip(reversed_input_ids, dims=(1,)).to(torch.cuda.current_device())  # (bsz, max_len)

        reversed_labels = [seq.flip(dims=(0,)) for seq in labels]
        reversed_labels = torch.nn.utils.rnn.pad_sequence(
            sequences=reversed_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )  # (bsz, max_len)
        
        labels = torch.flip(reversed_labels, dims=(1,)).to(torch.cuda.current_device())  # (bsz, max_len) 
        
        # Padding to model max length
        to_pad = self.model_max_length - input_ids.size(1)
        input_ids = F.pad(input_ids, (to_pad, 0), value=self.tokenizer.pad_token_id)
        labels = F.pad(labels, (to_pad, 0), value=IGNORE_INDEX)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(torch.cuda.current_device())
        outputs = self.model(input_ids, attention_mask=attention_mask)[0]

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        lens = (labels[..., 1:] != IGNORE_INDEX).sum(-1).cpu().numpy()

        loss_sum = loss.sum(-1).to(torch.float32).cpu().detach().numpy()
        return loss_sum.tolist(), lens.tolist()