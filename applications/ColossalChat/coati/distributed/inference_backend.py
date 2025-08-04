from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from colossalai.utils import get_current_device

from .utils import log_probs_from_logits, update_by_default

try:
    import sglang as sgl
except ImportError:
    sgl = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None


class BaseInferenceBackend:
    def __init__(self, model_config: Dict[str, Any], generate_config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        pass

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate new tokens given input_ids and attention_mask.

        Args:
            input_ids (torch.Tensor): shape [B, S]
            attention_mask (torch.Tensor): shape [B, S]

        Returns:
            Dict[str, torch.Tensor]: containing the
                - input_ids (torch.Tensor): shape [B, S+N]
                - attention_mask (torch.Tensor): shape [B, S+N]
                - action_log_probs (torch.Tensor): shape [B, N]
                - action_mask (torch.Tensor): shape [B, N]
                where N is the number of generated tokens. And all tensors should be on CUDA.
        """

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass


class TransformersInferenceBackend(BaseInferenceBackend):
    DEFAULT_MODEL_CONFIG = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    FORCE_MODEL_CONFIG = dict(
        device_map="auto",
    )
    FORCE_GENERATE_CONFIG = dict(output_logits=True, return_dict_in_generate=True)

    def __init__(
        self,
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        num_generations: int = 8,
        tokenizer_config: Dict[str, Any] = None,
    ):
        model_config = update_by_default(model_config, self.DEFAULT_MODEL_CONFIG)
        model_config.update(self.FORCE_MODEL_CONFIG)
        path = model_config.pop("path")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(path, **model_config)
        self.generate_config = generate_config.copy()
        self.generate_config.update(self.FORCE_GENERATE_CONFIG)
        self.tokenizer = tokenizer
        self.num_generations = num_generations

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        micro_batch_size = input_ids.size(0)
        input_ids = input_ids.to(get_current_device())
        attention_mask = attention_mask.to(get_current_device())
        gt_answer = kwargs.pop("gt_answer", None)
        test_cases = kwargs.pop("test_cases", None)
        if self.num_generations > 1:
            input_ids = input_ids.repeat_interleave(self.num_generations, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.num_generations, dim=0)
        out = self.model.generate(
            input_ids, attention_mask=attention_mask, **kwargs, **self.generate_config, tokenizer=self.tokenizer
        )
        input_len = input_ids.shape[-1]
        new_token_ids = out.sequences[:, input_len:]
        # get log probs
        assert new_token_ids.shape[-1] == len(out.logits)
        action_log_probs = []
        for i, logits in enumerate(out.logits):
            action_log_probs.append(log_probs_from_logits(logits[:, None, :], new_token_ids[:, i : i + 1]))
        action_log_probs = torch.cat(action_log_probs, dim=1)
        # get action mask
        response_idx = torch.zeros((new_token_ids.size(0), 2), dtype=torch.int).to(get_current_device())
        action_mask = torch.ones_like(new_token_ids, dtype=attention_mask.dtype)
        if self.tokenizer.eos_token_id is not None:
            for indices in torch.nonzero(new_token_ids == self.tokenizer.eos_token_id):
                action_mask[indices[0], indices[1] + 1 :] = 0
        response_idx[:, 0] = input_len
        response_idx[:, 1] = input_len + action_mask.sum(dim=1) - 1

        if attention_mask.size(0) != action_mask.size(0):
            assert action_mask.size(0) % attention_mask.size(0) == 0
            attention_mask = attention_mask.repeat_interleave(action_mask.size(0) // attention_mask.size(0), dim=0)

        attention_mask = torch.cat((attention_mask, action_mask), dim=1)
        data = {
            "input_ids": out.sequences,
            "attention_mask": attention_mask,
            "action_log_probs": action_log_probs,
            "action_mask": action_mask,
            "response_idx": response_idx,
        }

        data = {k: v.view(micro_batch_size, self.num_generations, v.size(-1)) for k, v in data.items()}

        if gt_answer is not None:
            data["gt_answer"] = gt_answer
        if test_cases is not None:
            data["test_cases"] = test_cases
        data = {k: v.to(get_current_device()) for k, v in data.items()}
        return data

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)


class SGLangInferenceBackend(BaseInferenceBackend):
    def __init__(
        self,
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        num_generations: int = 8,
        tokenizer_config: Dict[str, Any] = None,
    ):
        if sgl is None:
            raise ImportError("sglang is not installed")
        path = model_config.pop("path")
        defaut_config = dict(
            trust_remote_code=True,
            skip_tokenizer_init=True,
        )
        defaut_config.update(model_config)
        self.llm = sgl.Engine(model_path=path, **defaut_config)
        self.generate_config = generate_config
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(path)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.llm.generate(input_ids=input_ids.tolist(), sampling_params=self.generate_config)
        out_tokens = []
        out_len = []
        for out in outputs:
            out_tokens.append(out["token_ids"])
            out_len.append(out["meta_info"]["completion_tokens"])
        max_len = max(out_len)
        input_len = input_ids.shape[-1]
        attention_mask = F.pad(attention_mask, (0, max_len), value=1)
        for i in range(len(out_tokens)):
            out_tokens[i] = out_tokens[i] + [self.tokenizer.pad_token_id] * (max_len - out_len[i])
            attention_mask[i, input_len + out_len[i] :] = 0
        out = torch.tensor(out_tokens)
        out = torch.cat((input_ids, out), dim=1)
        labels = out.clone()
        labels[..., :input_len] = -100
        for i in range(len(out_len)):
            labels[i, input_len + out_len[i] :] = -100
        data = {
            "input_ids": out,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        data = {k: v.to(get_current_device()) for k, v in data.items()}
        return data

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if self.config.tie_word_embeddings:
            del state_dict["lm_head.weight"]
        named_tensors = [(k, v) for k, v in state_dict.items()]
        self.llm.update_weights_from_tensor(named_tensors)


class VLLMInferenceBackend(BaseInferenceBackend):
    DEFAULT_MODEL_CONFIG = dict(
        trust_remote_code=True,
        enable_sleep_mode=False,
    )
    FORCE_GENERATE_CONFIG = dict(
        logprobs=0,
    )

    def __init__(
        self,
        model_config: Dict[str, Any],
        generate_config: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        num_generations: int = 8,
        tokenizer_config: Dict[str, Any] = None,
    ):
        if LLM is None:
            raise ImportError("vllm is not installed")
        model_config = update_by_default(model_config, self.DEFAULT_MODEL_CONFIG)
        path = model_config.pop("path")
        tokenizer_path = tokenizer_config.get("path", None) if tokenizer_config is not None else None
        self.llm = LLM(model=path, tokenizer=tokenizer_path, **model_config)
        generate_config = generate_config.copy()
        generate_config.update(self.FORCE_GENERATE_CONFIG)
        generate_config.update({"n": num_generations})
        self.generate_config = generate_config
        self.sample_params = SamplingParams(**generate_config)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.num_generations = num_generations

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        micro_batch_size = input_ids.size(0)
        response_start_idx = input_ids.size(1)
        first_non_padding_token_idx = (input_ids != self.tokenizer.pad_token_id).int().argmax(dim=1)
        micro_batch_input_ids = input_ids.tolist()
        micro_batch_input_ids_no_padding = [
            micro_batch_input_ids[i][first_non_padding_token_idx[i] :] for i in range(micro_batch_size)
        ]
        sample_params = kwargs.get("sample_params", self.sample_params)
        outputs = self.llm.generate(
            prompt_token_ids=micro_batch_input_ids_no_padding, sampling_params=sample_params, use_tqdm=False
        )
        out_tokens = []
        out_len = []
        log_probs = []
        response_idx = []
        for out in outputs:
            for output_i in out.outputs:
                out_len.append(len(output_i.token_ids))
                out_tokens.append(list(output_i.token_ids))
                response_idx.append((response_start_idx, response_start_idx + len(output_i.token_ids) - 1))
                assert len(output_i.logprobs) == len(output_i.token_ids)
                p = [m[t].logprob for m, t in zip(output_i.logprobs, output_i.token_ids)]
                log_probs.append(p)

        # pad them
        max_len = self.sample_params.max_tokens
        action_mask = torch.ones(len(out_tokens), max_len, dtype=attention_mask.dtype)

        for i, new_token_ids in enumerate(out_tokens):
            pad_len = max_len - out_len[i]
            out_tokens[i] = new_token_ids + [self.tokenizer.pad_token_id] * pad_len
            log_probs[i] = log_probs[i] + [0.0] * pad_len
            action_mask[i, out_len[i] :] = 0

        out_tokens = torch.tensor(out_tokens)
        log_probs = torch.tensor(log_probs)
        response_idx = torch.tensor(response_idx)

        if attention_mask.size(0) != action_mask.size(0):
            assert action_mask.size(0) % attention_mask.size(0) == 0
            num_returns = action_mask.size(0) // attention_mask.size(0)
            attention_mask = attention_mask.repeat_interleave(num_returns, dim=0)
            input_ids = input_ids.repeat_interleave(num_returns, dim=0)

        out_tokens = torch.cat((input_ids, out_tokens), dim=1)
        attention_mask = torch.cat((attention_mask, action_mask), dim=1)

        data = {
            "input_ids": out_tokens,
            "attention_mask": attention_mask,
            "action_log_probs": log_probs,
            "action_mask": action_mask,
            "response_idx": response_idx,
        }

        data = {k: v.view(micro_batch_size, -1, v.size(-1)) for k, v in data.items()}
        data = {k: v.to(get_current_device()) for k, v in data.items()}
        if "gt_answer" in kwargs:
            data["gt_answer"] = kwargs["gt_answer"]
        if "test_cases" in kwargs:
            data["test_cases"] = kwargs["test_cases"]
        return data

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


BACKEND_MAP = {
    "transformers": TransformersInferenceBackend,
    # "sglang": SGLangInferenceBackend, # sglang backend will stuck the process due to unknown reason
    "vllm": VLLMInferenceBackend,
}
