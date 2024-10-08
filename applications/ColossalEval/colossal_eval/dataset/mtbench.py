import copy
import json
import os
from collections import defaultdict
from typing import Dict, List

from colossal_eval.utils import get_json_list

from colossalai.logging import DistributedLogger

from .base import BaseDataset

default_inference_kwargs = {
    "calculate_loss": False,
    "all_classes": None,
    "language": "English",
    "calculate_overall_loss": False,
    "max_new_tokens": 1024,
    "turns": 2,
}


class MTBenchDataset(BaseDataset):
    """
    Dataset class for mt_bench dataset.
    Data source: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl
    This dataset class will convert the original dataset into the inference dataset.
    """

    def __init__(self, path, logger: DistributedLogger, *args, **kwargs):
        self.multiturn = True
        self.dataset = self.load(path, logger, *args, **kwargs)

    @staticmethod
    def load(path: str, logger: DistributedLogger, *args, **kwargs) -> List[Dict]:
        dataset = {"test": defaultdict(dict)}

        file_path = os.path.join(path, "question.jsonl")
        ref_path = os.path.join(path, "reference_answer/gpt-4.jsonl")

        reference = defaultdict(list)
        ref_origin = get_json_list(ref_path)
        for ref in ref_origin:
            reference[ref["question_id"]] = ref["choices"][0]["turns"]

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                question = json.loads(line)
                category = question["category"]
                turn_number = len(question["turns"])
                data_point = {
                    "id": question["question_id"],
                    "dataset": "mtbench",
                    "split": "test",
                    "category": category,
                    "instruction": question["turns"],
                    "input": "",
                    "output": [],
                    "target": (
                        [""] * turn_number
                        if question["question_id"] not in reference
                        else reference[question["question_id"]]
                    ),
                }

                if category in dataset["test"]:
                    dataset["test"][category]["data"].append(data_point)
                else:
                    dataset["test"][category] = {
                        "data": [data_point],
                        "inference_kwargs": copy.deepcopy(default_inference_kwargs),
                    }

        return dataset
