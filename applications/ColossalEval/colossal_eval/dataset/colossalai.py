from collections import defaultdict
from copy import deepcopy
from typing import Dict, List

from colossal_eval.utils import jload

from colossalai.logging import DistributedLogger

from .base import BaseDataset

default_inference_kwargs = {
    "calculate_loss": False,
    "all_classes": None,
    "language": "Chinese",
    "calculate_overall_loss": False,
    "max_new_tokens": 256,
}

# You can add your own subcategory questions and specify whether it is a single-choice question or has target answers and need to calculate loss.
single_choice_question = set()
calculate_loss = set()


def get_data_per_category(data):
    data_per_category = defaultdict(list)
    for item in data:
        category = item["category"]
        data_per_category[category].append(item)

    return data_per_category


class ColossalDataset(BaseDataset):
    """
    Dataset class for Colossal dataset.
    This dataset class will convert the original dataset into the inference dataset.
    """

    @staticmethod
    def load(path: str, logger: DistributedLogger, *args, **kwargs) -> List[Dict]:
        dataset = {"test": {}}
        data = jload(path)
        data_per_category = get_data_per_category(data)
        categories = list(data_per_category.keys())

        for category in categories:
            dataset["test"][category] = {"data": []}
            category_data = data_per_category[category]

            dataset["test"][category]["inference_kwargs"] = deepcopy(default_inference_kwargs)

            if category in calculate_loss:
                dataset["test"][category]["inference_kwargs"]["calculate_loss"] = True
            if category in single_choice_question:
                dataset["test"][category]["inference_kwargs"]["all_classes"] = ["A", "B", "C", "D"]

            for item in category_data:
                data_sample = {
                    "dataset": "colossal",
                    "split": "test",
                    "category": category,
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "output": "",
                    "target": item["target"],
                    "id": item["id"],
                }
                dataset["test"][category]["data"].append(data_sample)

        return dataset
