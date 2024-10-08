import json
import os
import re
from copy import deepcopy
from typing import Dict, List

from colossalai.logging import DistributedLogger

from .base import BaseDataset

multi_choice_datasets = [
    "Chinese Lang and Usage MCQs",
    "Chinese Modern Lit",
    "English Fill in Blanks",
    "English Reading Comp",
    "Geography MCQs",
    "Physics MCQs",
    "English Cloze Test",
]

chinese_qa_datasets = [
    "Biology MCQs",
    "Chemistry MCQs",
    "Chinese Lang and Usage MCQs",
    "Chinese Modern Lit",
    "Geography MCQs",
    "History MCQs",
    "Math I MCQs",
    "Math II MCQs",
    "Physics MCQs",
    "Political Science MCQs",
]
english_qa_datasets = ["English MCQs", "English Fill in Blanks", "English Reading Comp", "English Cloze Test"]

default_inference_kwargs = {
    "calculate_loss": True,
    "all_classes": None,
    "language": "Chinese",
    "calculate_overall_loss": False,
    "max_new_tokens": 32,
}


def get_all_classes(instruction: str):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pattern = r"([A-Z]\. |[A-Z]．|[A-Z]\.)"
    options = sorted(list(set(re.findall(pattern, instruction))))
    options = sorted(list(set([string[0] for string in options])))

    for i in range(len(options)):
        if options[i] == letters[i]:
            continue
        else:
            return options[0:i]
    return options


class GaoKaoBenchDataset(BaseDataset):
    """
    Dataset class for GAOKAO-Bench dataset.
    Data source: https://github.com/OpenLMLab/GAOKAO-Bench/tree/main/data
    This dataset class will convert the original dataset into the inference dataset.

    A few typos needed to be manually corrected in the origin dataset, some of the following is fixed.
    Issue link: https://github.com/OpenLMLab/GAOKAO-Bench/issues/20
    1. Option C missing in index 111 in 2010-2022_Chemistry_MCQs.json
    2. Option B missing "." after it in index 16 in 2012-2022_English_Cloze_Test.json
    3. Option G missing "." after it in index 23 in 2012-2022_English_Cloze_Test.json
    """

    @staticmethod
    def load(path: str, logger: DistributedLogger, *args, **kwargs) -> List[Dict]:
        dataset = {"test": {}}
        for category in ["Fill-in-the-blank_Questions", "Multiple-choice_Questions", "Open-ended_Questions"]:
            files = os.listdir(os.path.join(path, "data", category))
            files.sort()

            for file in files:
                subject = file[10:-5].split("_")
                subject = " ".join(subject)
                dataset["test"][subject] = {"data": []}

                file_dir = os.path.join(path, "data", category, file)

                with open(file_dir, encoding="utf-8") as f:
                    data = json.load(f)

                    # It's been tested that each data sample in one subcategory have same inference arguments.
                    inference_kwargs = deepcopy(default_inference_kwargs)
                    if category == "Multiple-choice_Questions" and subject not in multi_choice_datasets:
                        all_classes = get_all_classes(data["example"][0]["question"])
                        inference_kwargs["all_classes"] = all_classes
                    if subject in english_qa_datasets:
                        inference_kwargs["language"] = "English"
                    if subject in chinese_qa_datasets:
                        inference_kwargs["language"] = "Chinese"

                    dataset["test"][subject]["inference_kwargs"] = inference_kwargs

                    for sample in data["example"]:
                        # Convert multi-choice answers to a single string.
                        # We will convert it back when evaluating.
                        # We do this because if target is a list, it should be only used for multiple target answers.
                        if subject in multi_choice_datasets:
                            sample["answer"] = "".join(sample["answer"])

                        if isinstance(sample["answer"], list) and len(sample["answer"]) == 1:
                            sample["answer"] = sample["answer"][0]

                        data_sample = {
                            "dataset": "gaokaobench",
                            "split": "test",
                            "category": f"{category[:-10]}-{subject}",
                            "instruction": sample["question"].strip() + "\n答案：",
                            "input": "",
                            "output": "",
                            "target": sample["answer"],
                        }

                        dataset["test"][subject]["data"].append(data_sample)

        return dataset
