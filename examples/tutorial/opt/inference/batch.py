from typing import Any, Deque, Hashable, List, Tuple

import torch
from energonai import BatchManager, SubmitEntry, TaskEntry


class BatchManagerForGeneration(BatchManager):
    def __init__(self, max_batch_size: int = 1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.pad_token_id = pad_token_id

    def _left_padding(self, batch_inputs):
        max_len = max(len(inputs["input_ids"]) for inputs in batch_inputs)
        outputs = {"input_ids": [], "attention_mask": []}
        for inputs in batch_inputs:
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            padding_len = max_len - len(input_ids)
            input_ids = [self.pad_token_id] * padding_len + input_ids
            attention_mask = [0] * padding_len + attention_mask
            outputs["input_ids"].append(input_ids)
            outputs["attention_mask"].append(attention_mask)
        for k in outputs:
            outputs[k] = torch.tensor(outputs[k])
        return outputs, max_len

    @staticmethod
    def _make_batch_key(entry: SubmitEntry) -> tuple:
        data = entry.data
        return (data["top_k"], data["top_p"], data["temperature"])

    def make_batch(self, q: Deque[SubmitEntry]) -> Tuple[TaskEntry, dict]:
        entry = q.popleft()
        uids = [entry.uid]
        batch = [entry.data]
        while len(batch) < self.max_batch_size:
            if len(q) == 0:
                break
            if self._make_batch_key(entry) != self._make_batch_key(q[0]):
                break
            if q[0].data["max_tokens"] > entry.data["max_tokens"]:
                break
            e = q.popleft()
            batch.append(e.data)
            uids.append(e.uid)
        inputs, max_len = self._left_padding(batch)
        trunc_lens = []
        for data in batch:
            trunc_lens.append(max_len + data["max_tokens"])
        inputs["top_k"] = entry.data["top_k"]
        inputs["top_p"] = entry.data["top_p"]
        inputs["temperature"] = entry.data["temperature"]
        inputs["max_tokens"] = max_len + entry.data["max_tokens"]
        return TaskEntry(tuple(uids), inputs), {"trunc_lens": trunc_lens}

    def split_batch(self, task_entry: TaskEntry, trunc_lens: List[int] = []) -> List[Tuple[Hashable, Any]]:
        retval = []
        for uid, output, trunc_len in zip(task_entry.uids, task_entry.batch, trunc_lens):
            retval.append((uid, output[:trunc_len]))
        return retval
