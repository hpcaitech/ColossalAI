from typing import Any, Dict

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizer

from colossalai.utils import get_current_device

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
        pass

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        pass


class TransformersInferenceBackend(BaseInferenceBackend):
    def __init__(self, model_config: Dict[str, Any], generate_config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        path = model_config.pop("path")
        defaut_config = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        defaut_config.update(model_config)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(path, **defaut_config)
        self.generate_config = generate_config

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        input_ids = input_ids.to(get_current_device())
        attention_mask = attention_mask.to(get_current_device())
        out = self.model.generate(input_ids, attention_mask=attention_mask, **kwargs, **self.generate_config)
        input_len = input_ids.shape[-1]
        labels = out.clone()
        labels[..., :input_len] = -100
        attention_mask = F.pad(attention_mask, (0, out.shape[-1] - input_len), value=1)
        attention_mask = attention_mask.expand_as(labels)
        data = {
            "input_ids": out,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)


class SGLangInferenceBackend(BaseInferenceBackend):
    def __init__(self, model_config: Dict[str, Any], generate_config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
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
    def __init__(self, model_config: Dict[str, Any], generate_config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        if LLM is None:
            raise ImportError("vllm is not installed")
        path = model_config.pop("path")
        defaut_config = dict(
            trust_remote_code=True,
            # skip_tokenizer_init=True,
        )
        defaut_config.update(model_config)
        self.llm = LLM(path, **defaut_config)
        self.generate_config = SamplingParams(**generate_config, stop_token_ids=[tokenizer.eos_token_id])
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(path)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.llm.generate(
            prompt_token_ids=input_ids.tolist(), sampling_params=self.generate_config, use_tqdm=False
        )
        out_tokens = []
        out_len = []
        for out in outputs:
            out_tokens.append(list(out.outputs[0].token_ids))
            out_len.append(len(out.outputs[0].token_ids))
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
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


BACKEND_MAP = {
    "transformers": TransformersInferenceBackend,
    "sglang": SGLangInferenceBackend,
    "vllm": VLLMInferenceBackend,
}
