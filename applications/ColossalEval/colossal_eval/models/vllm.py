import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from colossal_eval.utils import Conversation, get_batch_prompt, is_rank_0
from torch.utils.data import DataLoader
from tqdm import tqdm
from vllm import LLM, SamplingParams

from colossalai.logging import DistributedLogger

from .huggingface import HuggingFaceModel

IGNORE_INDEX = -100


class vLLMModel(HuggingFaceModel):
    """
    Model wrapper around vLLM models.

    Args:
        path: The path to a vLLM model.
        model_max_length: The maximum sequence length of the model.
        tokenizer_path: The path to the tokenizer.
        tokenizer_kwargs: Keyword arguments for the tokenizer.
        model_kwargs: Keyword arguments for the model.
        prompt_template: The model's prompt template.
        batch_size: Batch size for inference.
        logger: Logger for the model.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed execution with tensor parallelism.
        quantization: The method used to quantize the model weights
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading the model weights.
        enforce_eager: Whether to enforce eager execution.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
        disable_custom_all_reduce: See ParallelConfig
    """

    def __init__(
        self,
        path: str,
        model_max_length: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: Dict = None,
        model_kwargs: Dict = None,
        prompt_template: Conversation = None,
        batch_size: int = 1,
        logger: DistributedLogger = None,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.5,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            model_max_length=model_max_length,
            prompt_template=prompt_template,
            batch_size=batch_size,
            logger=logger,
        )

        self._load_model(
            path=path,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            tokenizer_path=tokenizer_path if tokenizer_path else None,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
        )

    def _load_model(
        self,
        path: str,
        model_kwargs: dict,
        tokenizer_kwargs: dict,
        tokenizer_path: Optional[str] = None,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
    ):
        """
        Load model.

        Args:
            path: The path to the model.
            model_kwargs: Keyword arguments for the model.
            tokenizer_kwargs: Keyword arguments for the tokenizer.
            tokenizer_path: The path to the tokenizer.
            trust_remote_code: Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.
            tensor_parallel_size: The number of GPUs to use for distributed execution with tensor parallelism.
            quantization: The method used to quantize the model weights
            gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache.
            swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            cpu_offload_gb: The size (GiB) of CPU memory to use for offloading the model weights.
            enforce_eager: Whether to enforce eager execution.
            max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            disable_custom_all_reduce: See ParallelConfig

        """
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = eval(model_kwargs["torch_dtype"])
            model_kwargs.pop("torch_dtype")
        else:
            model_kwargs.setdefault("dtype", torch.float16)

        if "trust_remote_code" in model_kwargs:
            trust_remote_code = model_kwargs["trust_remote_code"]
            model_kwargs.pop("trust_remote_code")

        if "trust_remote_code" in tokenizer_kwargs:
            trust_remote_code = tokenizer_kwargs["trust_remote_code"]
            tokenizer_kwargs.pop("trust_remote_code")

        self.model = LLM(
            model=path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **model_kwargs,
            **tokenizer_kwargs,
        )

        self.tokenizer = self.model.get_tokenizer()

        if self.batch_size > 1:
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.logger.warning("pad_token_id is not set for the tokenizer. " "Using eos_token_id as pad_token_id.")
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, "eod_id"):
                # Qwen has an eod token "<|endoftext|>".
                self.tokenizer.pad_token_id = self.tokenizer.eod_id
            else:
                self.logger.error("Neither eos_token nor eod_id is available for setting pad_token_id.")
                raise ValueError(
                    "The tokenizer does not have a pad_token_id, eos_token, or eod_id. "
                    "Please set pad_token_id manually."
                )

    def _calculate_loss(self, inputs: List[str], labels: List[str]) -> Tuple[List]:
        """
        Calculate loss on target tokens. Adapted from https://github.com/open-compass/opencompass/blob/c2bcd8725e615ec455bf5b7301f8d09962cd64e3/opencompass/models/vllm.py#L110

        Args:
            input_ids_list: A batch of input string.
            labels: A batch of labels.

        Returns:
            A list of loss and a list of label length.

        """
        batch_size = len(inputs)
        sampling_kwargs = SamplingParams(logprobs=1)
        outputs = self.model.generate(inputs, sampling_kwargs)
        ce_loss = []

        if labels is not None:
            lens = [len(self.tokenizer.encode(label, add_special_tokens=False)) for label in labels]
        else:
            lens = [1] * batch_size

        for i in range(batch_size):
            logprobs = outputs[i].outputs[0].logprobs
            token_ids = outputs[i].outputs[0].token_ids

            logprobs_list = [logprobs[i][token_ids[i]] for i in range(len(logprobs))]
            logprobs_list = [i.logprob for i in logprobs_list]
            logprobs_list = np.array(logprobs_list)

            if lens is not None:
                logprobs_list = logprobs_list[: lens[i]]

            loss = -logprobs_list.sum(axis=-1) / lens[i]
            ce_loss.append(loss)

        batch_loss = np.array(ce_loss)

        return batch_loss, lens

    def inference(self, data_loader: DataLoader, inference_kwargs: Dict[str, Any], debug: bool = False) -> List[Dict]:
        """
        Infer the given data.
        This function will call self.generate() to get model outputs and use LogitsProcessor param to get specific logits.

        Args:
            data: The data for inference.
            inference_kwargs: Arguments for inference.
            debug: Whether to display generated prompt for debugging.

        Returns:
            Inference results.

        """
        calculate_loss = inference_kwargs["calculate_loss"]
        classes = inference_kwargs["all_classes"]
        language = inference_kwargs["language"]
        calculate_overall_loss = inference_kwargs["calculate_overall_loss"]
        max_new_tokens = inference_kwargs["max_new_tokens"]
        few_shot_data = inference_kwargs.get("few_shot_data", None)

        # Some classification questions' options are texts not a single letter such as A, B, C and D.
        # If the text length is greater than 1, we won't calculate loss over choices.
        if classes is not None and any(len(c) > 1 for c in classes):
            classes = None

        self.choices = classes
        self.indices_for_choices = None
        if self.choices:
            # Get indices for each choice
            self._get_choices_indices(language)

            self.str_label_map = {choice: idx for idx, choice in enumerate(self.choices)}

        bar = tqdm(
            range(len(data_loader)),
            desc=f"{inference_kwargs['dataset']}-{inference_kwargs['category']} Inference steps",
            disable=not is_rank_0(),
        )
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        answers = []

        for i, batch in enumerate(data_loader):
            batch_prompt, batch_target = get_batch_prompt(
                self.prompt_template, batch, few_shot_data, self.tokenizer, self.model_max_length
            )

            if is_rank_0() and debug and i == 0:
                self.logger.info(
                    f"Inference arguments for dataset {batch[0]['dataset']} category {batch[0]['category']} is:\n{inference_kwargs}"
                )
                self.logger.info("-" * 120)
                self.logger.info("An example prompt and prompt with target is:")
                self.logger.info("-" * 120)
                self.logger.info(batch_prompt[0])
                self.logger.info("-" * 120)
                self.logger.info(batch_prompt[0] + batch_target[0][0])

            if not calculate_overall_loss:
                batch_decodes, scores = self.generate(batch_prompt, max_new_tokens)

            if calculate_loss:
                batch_losses, batch_target_token_nums, batch_bytes_nums = self.get_loss(
                    batch_prompt, batch_target, calculate_overall_loss
                )

            probs = []
            if self.indices_for_choices:
                scores = scores.to(torch.float32)
                # If we have indices_for_choices(must be single-choice question), there will be only one target answer for one data sample.
                # Otherwise this will violate the single-choice setting.

                if calculate_loss:
                    labels = [self.str_label_map[batch[j]["target"]] for j in range(len(batch))]

                    loss_over_choices = loss_fct(scores, torch.tensor(labels, dtype=torch.long)).numpy().tolist()

                probs = scores.numpy().tolist()
                probs = [
                    {choice: probs[i][self.str_label_map[choice]] for choice in self.choices} for i in range(len(probs))
                ]

            for j in range(len(batch)):
                if not calculate_overall_loss:
                    if isinstance(batch[j]["output"], list):
                        batch[j]["output"].append(batch_decodes[j].strip())
                    else:
                        batch[j]["output"] = batch_decodes[j].strip()

                    if isinstance(scores, torch.Tensor):
                        batch[j]["logits_over_choices"] = probs[j]

                        if calculate_loss:
                            batch[j]["loss_over_choices"] = loss_over_choices[j]

                if calculate_loss:
                    batch[j]["loss"] = (np.array(batch_losses[j]) / np.array(batch_target_token_nums[j])).tolist()

                    # loss_sum is specially used for pertrain dataset for calculating per-byte-perplexity.
                    # However, loss (which is per sample loss) suffices for most cases.
                    batch[j]["loss_sum"] = batch_losses[j]
                    batch[j]["token_num"] = batch_target_token_nums[j]

                    if batch_bytes_nums:
                        batch[j]["byte_num"] = batch_bytes_nums[j]
            answers.extend(batch)

            bar.update()

        return answers

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

        generation_kwargs = kwargs.copy()
        generation_kwargs.update({"max_tokens": max_new_tokens})
        logits_processor = GetTokenLogitsProcessor(self.indices_for_choices)

        sampling_kwargs = SamplingParams(logits_processors=[logits_processor], **generation_kwargs)

        outputs = self.model.generate(truncated_inputs, sampling_kwargs)
        output_strs = []
        for output in outputs:
            generated_text = output.outputs[0].text
            output_strs.append(generated_text)
        scores = logits_processor.get_target_logits()

        return output_strs, scores

    @torch.no_grad()
    def get_loss(
        self, batch_prompt: List[str], batch_target: List[List[str]], calculate_overall_loss: bool
    ) -> List[List[float]]:
        """
        Calculate loss only on target tokens.

        Args:
            batch: A batch of prompt without target answer.
            batch_target: A batch of target answer. Sometimes one question can have multiple target answers.

        Returns:
            Loss.

        """

        # We set max_new_tokens in self._get_truncated_prompts to 0 because we only need logits to calculate loss.
        # We don't need to generate new tokens.
        # Target answer's length is usually << model_max_length, but we still call it in case.
        # We don't call self._get_truncated_prompts for batch_prompt because we need target answer's length first to reserve some space for target answer's tokens.
        if not calculate_overall_loss:
            batch_target = [self._get_truncated_prompts(prompt_target, 0) for prompt_target in batch_target]

        # Get the number of target answers for different questions
        batch_target_nums = [len(prompt_target) for prompt_target in batch_target]

        if calculate_overall_loss:
            batch = []
            bytes_list = []
            batch_prompt_pretrain = []
            for p, b in zip(batch_prompt, batch_target):
                batch.append(p + b[0])

            for input in batch:
                # Pretrain data tends to be very long, sometimes much larger than the model_max_length, we only tokenize 1/ratio of the data first to accelerate the tokenization process.
                # Once the length of the result is greater or equal to model_max_length, we stop iterating on ratios and use the result as input_ids and labels.
                # After all, the rest of the original string doesn't need to be tokenized at the first place.
                # Pretrain data tends to be very long, sometimes much larger than the model_max_length, we only tokenize 1/ratio of the data first to accelerate the tokenization process.
                # Once the length of the result is greater or equal to model_max_length, we stop iterating on ratios and use the result as input_ids and labels.
                # After all, the rest of the original string doesn't need to be tokenized at the first place.
                ratio = [16, 8, 4, 2, 1]
                tokenized = None
                for r in ratio:
                    tokenized = self.tokenizer(
                        [input[0 : len(input) // r]],
                        truncation=True,
                        max_length=self.model_max_length,
                        return_tensors="pt",
                    )
                    if tokenized.input_ids.size(1) >= self.model_max_length:
                        break

                string = self.tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)
                batch_prompt_pretrain.append(string)
                bytes_list.append(len(string.encode("utf-8")))

            batch_prompt = copy.deepcopy(batch_prompt_pretrain)
            batch_target = None
        else:
            batch_prompt_processed = []
            batch_target_processed = []
            for prompt, targets in zip(batch_prompt, batch_target):
                for target in targets:
                    target_tokenized = self.tokenizer(
                        [target], truncation=True, max_length=self.model_max_length, return_tensors="pt"
                    )
                    max_new_tokens = target_tokenized["input_ids"][0].size(0)
                    prompt_with_correct_length = self._get_truncated_prompts([prompt], max_new_tokens)[0]
                    batch_prompt_processed.append(prompt_with_correct_length)
                    batch_target_processed.append(target)

            batch_prompt = copy.deepcopy(batch_prompt_processed)
            batch_target = copy.deepcopy(batch_target_processed)
            bytes_list = None

        # Because of multiple target answers, the final batch size may be greater than self.batch_size.
        # We will generate new batches.
        losses = []
        target_token_nums = []

        losses_per_batch, target_token_num_per_batch = self._calculate_loss(batch_prompt, batch_target)
        losses.extend(losses_per_batch)
        target_token_nums.extend(target_token_num_per_batch)

        start_indice = 0
        losses_per_sample = []

        target_token_nums_per_sample = []
        bytes_nums_per_sample = []
        for length in batch_target_nums:
            losses_per_sample.append(losses[start_indice : start_indice + length])
            target_token_nums_per_sample.append(target_token_nums[start_indice : start_indice + length])

            if bytes_list:
                bytes_nums_per_sample.append(bytes_list[start_indice : start_indice + length])

            start_indice += length

        if bytes_list:
            return losses_per_sample, target_token_nums_per_sample, bytes_nums_per_sample

        return losses_per_sample, target_token_nums_per_sample, None


class GetTokenLogitsProcessor:
    """
    LogitsProcessor to get specific logits

    Args:
        indices_for_choices: token indices of required tokens
        target_logits: store all the target logits
    """

    def __init__(
        self,
        indices_for_choices: List[List[int]],
    ):
        self.indices_for_choices = (indices_for_choices,)
        self.target_logits = []

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        choice_scores = []

        if not input_ids:
            for option_indices in self.indices_for_choices[0]:
                choice_scores.append(logits[option_indices].detach().cpu())

            choice_scores = torch.max(torch.stack(choice_scores), dim=0)[0]
            self.target_logits.append(choice_scores)

        return logits

    def get_target_logits(self) -> torch.Tensor:
        return torch.stack(self.target_logits) if self.target_logits else torch.tensor([])
