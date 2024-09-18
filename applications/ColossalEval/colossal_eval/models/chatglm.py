import copy
from typing import List

import torch

from colossalai.utils import get_current_device

from .huggingface import HuggingFaceModel

IGNORE_INDEX = -100


class ChatGLMModel(HuggingFaceModel):
    def _get_truncated_prompts(self, inputs: List[str], max_new_tokens: int) -> List[str]:
        truncated_inputs = copy.deepcopy(inputs)
        # Adapted from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py#L187
        for i, input in enumerate(inputs):
            a_ids = self.tokenizer.encode(text=input, truncation=False, add_special_tokens=False)

            if len(a_ids) > self.model_max_length - max_new_tokens:
                half = (self.model_max_length - max_new_tokens) // 2
                prompt = self.tokenizer.decode(a_ids[:half], skip_special_tokens=True) + self.tokenizer.decode(
                    a_ids[-half:], skip_special_tokens=True
                )
                truncated_inputs[i] = prompt

        return truncated_inputs

    @torch.no_grad()
    def get_loss(
        self, batch_prompt: List[str], batch_target: List[List[str]], calculate_overall_loss: bool = False
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
        batch_target = [self._get_truncated_prompts(prompt_target, 0) for prompt_target in batch_target]

        # Get the number of target answers for different questions
        batch_target_nums = [len(prompt_target) for prompt_target in batch_target]

        labels_list = []
        input_ids_list = []

        for input, targets in zip(batch_prompt, batch_target):
            for target in targets:
                # Adapted from https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py#L187
                # If there is no history, the prompt is just the query.
                # We don't need to override self.generate() in ChatGLM-6B but need to override it in ChatGLM2-6B.
                # See https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1276
                target_tokenized = self.tokenizer.encode(text=target, add_special_tokens=False)

                # Get prompt with length model_max_length - len(target_tokenized).
                # Reserve some space for target answer tokens using max_new_tokens.
                # This will generate the correct start_idx and end_idx.
                max_new_tokens = len(target_tokenized)

                # Here 3 tokens are reserved for [gmask_id, bos_token, eos_id]. So we reserve max_new_tokens + 3 tokens.
                # See https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py#L323
                prompt_with_correct_length = self._get_truncated_prompts([input], max_new_tokens + 3)[0]
                input_tokenized = self.tokenizer.encode(prompt_with_correct_length, add_special_tokens=False)

                input_ids = self.tokenizer.build_inputs_with_special_tokens(input_tokenized, target_tokenized)

                context_length = input_ids.index(self.tokenizer.bos_token_id)
                context_length - 1

                target_ids = [IGNORE_INDEX] * len(input_ids)

                # -1 is for eos_token, we don't want to calculate loss on eos token.
                target_ids[-max_new_tokens - 1 : -1] = input_ids[-max_new_tokens - 1 : -1]

                input_ids_list.append(torch.LongTensor(input_ids))
                labels_list.append(torch.LongTensor(target_ids))

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
        for length in batch_target_nums:
            losses_per_sample.append(losses[start_indice : start_indice + length])
            target_token_nums_per_sample.append(target_token_nums[start_indice : start_indice + length])
            start_indice += length

        return losses_per_sample, target_token_nums_per_sample, None

    def _calculate_loss(self, input_ids_list: List[torch.LongTensor], labels: List[torch.LongTensor]) -> List[float]:
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
        ).to(get_current_device())
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to(
            get_current_device()
        )

        outputs = self.model(input_ids)[0]

        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_labels.size())

        lens = (labels != IGNORE_INDEX).sum(-1).cpu().numpy()

        loss_sum = loss.sum(-1).to(torch.float32).cpu().detach().numpy()
        return loss_sum.tolist(), lens.tolist()


class ChatGLM2Model(ChatGLMModel):
    def _get_truncated_prompts(self, inputs: List[str], max_new_tokens: int) -> List[str]:
        truncated_inputs = copy.deepcopy(inputs)
        # Adapted from https://github.com/THUDM/ChatGLM2-6B/blob/main/ptuning/main.py#L180
        for i, input in enumerate(inputs):
            a_ids = self.tokenizer.encode(text=input, add_special_tokens=True, truncation=False)

            if len(a_ids) > self.model_max_length - max_new_tokens:
                half = (self.model_max_length - max_new_tokens) // 2
                prompt = self.tokenizer.decode(a_ids[:half], skip_special_tokens=True) + self.tokenizer.decode(
                    a_ids[-half:], skip_special_tokens=True
                )
                truncated_inputs[i] = prompt

        return truncated_inputs

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
        # Follow the process of model.chat() method in modeling_chatglm2.py
        # See https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1020
        # See https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L1001

        query = []
        for input in inputs:
            prompt = self.tokenizer.build_prompt(input, None)
            query.append(prompt)

        truncated_query = self._get_truncated_prompts(query, max_new_tokens)

        encoded_inputs = self.tokenizer(
            truncated_query,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.model_max_length - max_new_tokens,
        ).to(get_current_device())

        # Set output_scores=True to get prediction scores.
        outputs = self.model.generate(
            **encoded_inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, **kwargs
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
    def get_loss(
        self, batch_prompt: List[str], batch_target: List[List[str]], calculate_overall_loss: bool = False
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
        batch_target = [self._get_truncated_prompts(prompt_target, 0) for prompt_target in batch_target]

        # Get the number of target answers for different questions
        batch_target_nums = [len(prompt_target) for prompt_target in batch_target]

        labels_list = []
        input_ids_list = []

        for input, targets in zip(batch_prompt, batch_target):
            for target in targets:
                # Adapted from https://github.com/THUDM/ChatGLM2-6B/blob/main/ptuning/main.py#L180
                prompt = self.tokenizer.build_prompt(input, None)

                target_tokenized = self.tokenizer.encode(
                    text=target, add_special_tokens=False, truncation=True, max_length=self.model_max_length
                )

                max_new_tokens = len(target_tokenized)
                prompt_with_correct_length = self._get_truncated_prompts([prompt], max_new_tokens)[0]
                input_tokenized = self.tokenizer.encode(
                    prompt_with_correct_length,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.model_max_length,
                )

                input_ids = input_tokenized + target_tokenized + [self.tokenizer.eos_token_id]
                target_ids = [IGNORE_INDEX] * len(input_ids)

                # -1 is for "eos"
                target_ids[-max_new_tokens - 1 : -1] = input_ids[-max_new_tokens - 1 : -1]

                input_ids_list.append(torch.LongTensor(input_ids))
                labels_list.append(torch.LongTensor(target_ids))

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
        for length in batch_target_nums:
            losses_per_sample.append(losses[start_indice : start_indice + length])
            target_token_nums_per_sample.append(target_token_nums[start_indice : start_indice + length])
            start_indice += length

        return losses_per_sample, target_token_nums_per_sample, None
