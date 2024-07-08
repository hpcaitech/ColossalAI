# Adapted from https://github.com/ModelTC/lightllm

from typing import Dict, List, Tuple

from .sampling_params import SamplingParams


class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, prompts: str = ""):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.prompts = prompts

    def to_rpc_obj(self):
        return {
            "request_id": self.request_id,
            "input_id": self.prompt_ids,
            "output_len": self.max_output_len,
            "sampling_param": self.sample_params.to_dict(),
        }

    def stop_sequences_matched(self):
        # should we add stpp sequences to the sample params?
        if self.sample_params.stop_sequences is not None:
            for stop_token_ids in self.sample_params.stop_sequences:
                stop_len = len(stop_token_ids)
                if (
                    stop_len > 0
                    and len(self.output_ids) >= stop_len
                    and all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len))
                ):
                    return True
        return False

    def __repr__(self):
        return f"request_id(n={self.request_id}, " f"prompt_ids={self.prompt_ids}, "


class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len
        return tokens

    def calcu_used_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + len(req.output_ids)
        return tokens

    def mark_finished_req(self, eos_id, engine_max_output_len):
        has_new_finish = False
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= engine_max_output_len:
                req.has_generate_finished = True
                has_new_finish = True
            if req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
        return has_new_finish

    def filter_finished(self) -> List[Req]:
        """
        Filter finished requests from the batch, the finished ones will be removed from 'reqs'.
        """
        # TODO: the logic of return should be defined here.
        unfinished_req = []
        finished_req = []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
            else:
                finished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return finished_req

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return f"batch_id={self.batch_id}, " f"reqs={self.reqs}, "

    def __len__(self):
        return len(self.reqs)


class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = (
            []
        )  # [req_id, new_token_id, gen_metadata, finished_state, abort_state]


class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = (
            []
        )  # [req_id, token_str, gen_metadata, finished_state, abort_state]


class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"outputs={self.outputs}, "
        )
