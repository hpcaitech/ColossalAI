import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from colossal_eval.utils import Conversation, get_batch_prompt, is_rank_0
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from colossalai.logging import DistributedLogger
from colossalai.shardformer import ShardConfig, ShardFormer

from .base import BaseModel

IGNORE_INDEX = -100


class HuggingFaceModel(BaseModel):
    """
    Model wrapper around HuggingFace AutoModel models.

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

    def __init__(
        self,
        path: str,
        model_max_length: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        model_kwargs: Dict = None,
        prompt_template: Conversation = None,
        batch_size: int = 1,
        logger: DistributedLogger = None,
        shard_config: ShardConfig = None,
    ):
        super().__init__(
            path=path,
            model_max_length=model_max_length,
            prompt_template=prompt_template,
            batch_size=batch_size,
            logger=logger,
        )
        self._load_tokenizer(path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs)

        self._load_model(path=path, model_kwargs=model_kwargs, peft_path=peft_path, shard_config=shard_config)

    def _get_choices_indices(self, language: str):
        """
        Get indices for each choice

        Some tokenizer will insert BOS if you don't specify add_special_tokens=False such as Llama-2.
        The indices for choices may be different given the context. For example, for Llama-2 tokenizer, for Chinese context like "答案：{choice}", indices for choices A, B, C and D are 29909, 29933, 29907 and 29928, for English context like "Answer: {choice}", indices for choices A, B, C and D are 319, 350, 315 and 360.
        print(self.tokenizer("答案：A")) to see
        print(self.tokenizer("Answer: A")) to see

        """

        # A trick for get "all" tokens ids related to given choices.
        self.indices_for_choices = [[] for _ in range(2)]
        for choice in self.choices:
            self.indices_for_choices[0].append(
                self.tokenizer(f"Answer: {choice}", add_special_tokens=False).input_ids[-1]
            )
            self.indices_for_choices[1].append(self.tokenizer(f"答案：{choice}", add_special_tokens=False).input_ids[-1])

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        """
        Load tokenizer.

        Args:
            path: The path to the model. Usually it also serves as the path to the tokenizer.
            tokenizer_path: The path to the tokenzier.
            tokenizer_kwargs: Keyword arguments for the tokenizer.

        """

        if self.batch_size > 1:
            tokenizer_kwargs.update({"padding_side": "left"})
            tokenizer_kwargs.update({"truncation_side": "left"})

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

        if self.tokenizer.pad_token_id is None:
            self.logger.warning("pad_token_id is not set for the tokenizer. " "Using eos_token_id as pad_token_id.")
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif hasattr(self.tokenizer, "eod_id"):
                # Qwen has an eod token "<|endoftext|>".
                self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def _load_model(
        self, path: str, model_kwargs: dict, peft_path: Optional[str] = None, shard_config: ShardConfig = None
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

        if shard_config is not None:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)
            shard_former = ShardFormer(shard_config)
            self.model, sharded_parameters = shard_former.optimize(self.model)
            self.model.to(torch.cuda.current_device())

            if peft_path is not None:
                raise NotImplementedError("ShardFormer for PEFT models is not implemented.")
        else:
            self.model = AutoModel.from_pretrained(path, **model_kwargs).to(torch.cuda.current_device())
            if peft_path is not None:
                self.model = PeftModel.from_pretrained(self.model, peft_path, is_trainable=False)
        self.model.eval()

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
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(torch.cuda.current_device())
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to(
            torch.cuda.current_device()
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(torch.cuda.current_device())

        outputs = self.model(input_ids, attention_mask=attention_mask)[0]

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        lens = (labels[..., 1:] != IGNORE_INDEX).sum(-1).cpu().numpy()

        loss_sum = loss.sum(-1).to(torch.float32).cpu().detach().numpy()
        return loss_sum.tolist(), lens.tolist()

    def _get_truncated_prompts(self, inputs: List[str], max_new_tokens: int) -> List[str]:
        """
        Truncate the input sequence to fit model_max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        https://github.com/THUDM/LongBench/blob/main/pred.py#L16

        Args:
            inputs: A batch of input prompts.
            max_new_tokens: Max new tokens for model to generate.

        Returns:
            Truncated prompts.

        """

        truncated_inputs = copy.deepcopy(inputs)
        for i, input in enumerate(inputs):
            tokenized_prompt = self.tokenizer(input, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > self.model_max_length - max_new_tokens:
                half = (self.model_max_length - max_new_tokens) // 2
                prompt = self.tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True
                ) + self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                truncated_inputs[i] = prompt

        return truncated_inputs

    def _get_input_ids_and_labels_pretrain(self, batch_prompt: List[str]) -> Tuple[List[torch.LongTensor]]:
        """
        Get input_ids and labels for pretrain data.
        We only need batch_prompt because for pretain dataset, we don't need to predict new tokens.

        Args:
            batch_prompt: A batch of prompt.

        Returns:
            Input_ids and labels for the given batch.

        """
        input_ids_list = []
        labels_list = []
        bytes_list = []

        for input in batch_prompt:
            # Pretrain data tends to be very long, sometimes much larger than the model_max_length, we only tokenize 1/ratio of the data first to accelerate the tokenization process.
            # Once the length of the result is greater or equal to model_max_length, we stop iterating on ratios and use the result as input_ids and labels.
            # After all, the rest of the original string doesn't need to be tokenized at the first place.
            ratio = [16, 8, 4, 2, 1]
            tokenized = None
            for r in ratio:
                tokenized = self.tokenizer(
                    [input[0 : len(input) // r]], truncation=True, max_length=self.model_max_length, return_tensors="pt"
                )
                if tokenized.input_ids.size(1) >= self.model_max_length:
                    break

            input_ids = copy.deepcopy(tokenized["input_ids"])[0]
            target_ids = copy.deepcopy(input_ids)

            string = self.tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=True)

            bytes_list.append(len(string.encode("utf-8")))

            input_ids_list.append(input_ids)
            labels_list.append(target_ids)

        return input_ids_list, labels_list, bytes_list

    def _get_input_ids_and_labels(
        self, batch_prompt: List[str], batch_target: List[List[str]], pretrain: bool
    ) -> Tuple[List[torch.LongTensor]]:
        """
        Get input_ids and labels for the given data.

        Args:
            batch_prompt: A batch of prompt.
            batch_target: A batch of target.

        Returns:
            Input_ids and labels for the given batch.

        """
        if pretrain:
            batch = []
            # Concatenate prompt and target answers.
            # You should decide the concatenation character in the corresponding dataset script in dataset folder. For example, in line 119 dataset/gsm.py, the concatenation character is space.
            for p, b in zip(batch_prompt, batch_target):
                batch.append(p + b[0])

            return self._get_input_ids_and_labels_pretrain(batch)

        input_ids_list = []
        labels_list = []

        for input, targets in zip(batch_prompt, batch_target):
            for target in targets:
                # TODO: Improve the labeling process. Should annotate the border by adding special tokens.
                target_tokenized = self.tokenizer(
                    [target], truncation=True, max_length=self.model_max_length, return_tensors="pt"
                )

                # Get prompt with length model_max_length - len(target_tokenized).
                # Reserve some space for target answer tokens using max_new_tokens.
                # This will generate the correct start_idx and end_idx.
                max_new_tokens = target_tokenized["input_ids"][0].size(0)
                prompt_with_correct_length = self._get_truncated_prompts([input], max_new_tokens)[0]
                input_tokenized = self.tokenizer(
                    [prompt_with_correct_length],
                    truncation=True,
                    max_length=self.model_max_length - max_new_tokens,
                    return_tensors="pt",
                )

                target_tokenized = self.tokenizer(
                    [prompt_with_correct_length + target],
                    truncation=True,
                    max_length=self.model_max_length,
                    return_tensors="pt",
                )

                start_idx = input_tokenized["input_ids"][0].size(0)
                end_idx = target_tokenized["input_ids"][0].size(0)

                # Sometimes if the target is only an option such as A, B, C and D, the length of input_tokenized is equal to the length of target_tokenized, so we need -1.
                # This is caused by the different behavior of tokenizers.
                # For example, the tokenizer for Baichuan and Llama will cause such problem in a plain prompt setting.
                # The length of the tokenized sequences for prompt "Answer: " and "Answer: A" is the same.
                # Baichuan: [29394, 31143, 31106] [29394, 31143, 703]
                # Llama: [673, 29901, 29871] [673, 29901, 319]
                # The length for sequence "prompt" and "prompt + A" is equal.
                # For ChatGLM, the length of the tokenized sequences is different.
                # ChatGLM: [16583, 12] [16583, 12, 167]

                if start_idx == end_idx:
                    start_idx -= 1

                input_ids = copy.deepcopy(target_tokenized["input_ids"])[0]
                target_ids = copy.deepcopy(input_ids)

                mask = torch.zeros_like(target_ids, dtype=torch.bool)
                mask[start_idx:end_idx] = True

                target_ids[~mask] = IGNORE_INDEX

                input_ids_list.append(input_ids)
                labels_list.append(target_ids)

        return input_ids_list, labels_list, None

    def inference(self, data: List[Dict], inference_kwargs: Dict[str, Any], debug: bool = False) -> List[Dict]:
        """
        Infer the given data.
        This function will call self.generate() to get model outputs and also self.model() to get logits.

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
        pretrain = inference_kwargs["pretrain"]
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

        turn = 0 if not isinstance(data[0]["output"], list) else len(data[0]["output"]) + 1
        turn_desc = "" if turn == 0 else f"-turn{turn}"

        bar = tqdm(
            range(math.ceil(len(data) / self.batch_size)),
            desc=f"{data[0]['dataset']}-{data[0]['category']}{turn_desc} Inference steps",
            disable=not is_rank_0(),
        )
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        answers = copy.deepcopy(data)
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            batch_prompt, batch_target = get_batch_prompt(
                self.prompt_template, batch, few_shot_data, self.tokenizer, language, self.model_max_length
            )

            if is_rank_0() and debug and i == 0:
                self.logger.info(
                    f"Inference arguments for dataset {data[0]['dataset']} category {data[0]['category']} is:\n{inference_kwargs}"
                )
                self.logger.info("-" * 120)
                self.logger.info("An example prompt and prompt with target is:")
                self.logger.info("-" * 120)
                self.logger.info(batch_prompt[0])
                self.logger.info("-" * 120)
                self.logger.info(batch_prompt[0] + batch_target[0][0])

            if not pretrain:
                batch_decodes, scores = self.generate(batch_prompt, max_new_tokens)

            if calculate_loss:
                batch_losses, batch_target_token_nums, batch_bytes_nums = self.get_loss(
                    batch_prompt, batch_target, pretrain
                )

            probs = []
            if self.indices_for_choices:
                scores = scores.to(torch.float32)
                # If we have indices_for_choices(must be single-choice question), there will be only one target answer for one data sample.
                # Otherwise this will violate the single-choice setting.

                if calculate_loss:
                    labels = [self.str_label_map[answers[i + j]["target"]] for j in range(len(batch_decodes))]

                    loss_over_choices = loss_fct(scores, torch.tensor(labels, dtype=torch.long)).numpy().tolist()

                probs = scores.numpy().tolist()
                probs = [
                    {choice: probs[i][self.str_label_map[choice]] for choice in self.choices} for i in range(len(probs))
                ]

            for j in range(len(batch_prompt)):
                if not pretrain:
                    if isinstance(answers[i + j]["output"], list):
                        answers[i + j]["output"].append(batch_decodes[j].strip())
                    else:
                        answers[i + j]["output"] = batch_decodes[j].strip()

                    if isinstance(scores, torch.Tensor):
                        answers[i + j]["logits_over_choices"] = probs[j]

                        if calculate_loss:
                            answers[i + j]["loss_over_choices"] = loss_over_choices[j]

                if calculate_loss:
                    answers[i + j]["loss"] = (np.array(batch_losses[j]) / np.array(batch_target_token_nums[j])).tolist()

                    # loss_sum is specially used for pertrain dataset for calculating per-byte-perplexity.
                    # However, loss (which is per sample loss) suffices for most cases.
                    answers[i + j]["loss_sum"] = batch_losses[j]
                    answers[i + j]["token_num"] = batch_target_token_nums[j]

                    if batch_bytes_nums:
                        answers[i + j]["byte_num"] = batch_bytes_nums[j]

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

        encoded_inputs = self.tokenizer(
            truncated_inputs,
            padding=True,
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

    @torch.no_grad()
    def get_loss(self, batch_prompt: List[str], batch_target: List[List[str]], pretrain: bool) -> List[List[float]]:
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
        if not pretrain:
            batch_target = [self._get_truncated_prompts(prompt_target, 0) for prompt_target in batch_target]

        # Get the number of target answers for different questions
        batch_target_nums = [len(prompt_target) for prompt_target in batch_target]

        input_ids_list, labels_list, bytes_list = self._get_input_ids_and_labels(batch_prompt, batch_target, pretrain)

        # Because of multiple target answers, the final batch size may be greater than self.batch_size.
        # We will generate new batches.
        losses = []
        target_token_nums = []

        batched_input_ids = [
            input_ids_list[i : i + self.batch_size] for i in range(0, len(input_ids_list), self.batch_size)
        ]
        batched_labels = [labels_list[i : i + self.batch_size] for i in range(0, len(labels_list), self.batch_size)]

        for batch_input_ids, batch_labels in zip(batched_input_ids, batched_labels):
            losses_per_batch, target_token_num_per_batch = self._calculate_loss(batch_input_ids, batch_labels)
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


class HuggingFaceCausalLM(HuggingFaceModel):
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
        self, path: str, model_kwargs: dict, peft_path: Optional[str] = None, shard_config: ShardConfig = None
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

        if shard_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            shard_former = ShardFormer(shard_config)
            self.model, sharded_parameters = shard_former.optimize(self.model)
            self.model.to(torch.cuda.current_device())

            if peft_path is not None:
                raise NotImplementedError("ShardFormer for PEFT models is not implemented.")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).to(torch.cuda.current_device())
            if peft_path is not None:
                self.model = PeftModel.from_pretrained(self.model, peft_path, is_trainable=False)

        self.model.eval()
