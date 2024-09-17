import copy
import csv
import os
from typing import Dict, List

from colossalai.logging import DistributedLogger

from .base import BaseDataset

default_inference_kwargs = {
    "calculate_loss": True,
    "all_classes": ["A", "B", "C", "D"],
    "language": "English",
    "calculate_overall_loss": False,
    "max_new_tokens": 32,
}


def get_few_shot_data(data: List[Dict], subject):
    few_shot_data = [f"The following are multiple choice questions (with answers) about {subject}."]
    for i in data:
        few_shot_data.append(i["input"] + i["target"])
    return few_shot_data


class MMLUDataset(BaseDataset):
    """
    Dataset class for MMLU dataset.
    Data source: https://github.com/hendrycks/test
    This dataset class will convert the original dataset into the inference dataset.
    """

    @staticmethod
    def load(path: str, logger: DistributedLogger, few_shot: bool, *args, **kwargs) -> List[Dict]:
        dataset = {"dev": {}, "test": {}}
        for split in ["dev", "test"]:
            files = os.listdir(os.path.join(path, split))
            files.sort()

            for file in files:
                subject = file[0 : -len(f"_{split}.csv")].split("_")
                subject = " ".join([word.title() if word != "us" else "US" for word in subject])

                file_dir = os.path.join(path, split, file)

                dataset[split][subject] = {"data": [], "inference_kwargs": {}}

                # It's been tested that each data sample in one subcategory have same inference arguments.
                dataset[split][subject]["inference_kwargs"] = copy.deepcopy(default_inference_kwargs)

                if split == "test" and few_shot:
                    dataset[split][subject]["inference_kwargs"]["few_shot_data"] = get_few_shot_data(
                        dataset["dev"][subject]["data"], subject
                    )

                with open(file_dir, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        assert len(row) == 6
                        choices = f"A. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                        data_sample = {
                            "dataset": "mmlu",
                            "split": split,
                            "category": subject,
                            "instruction": f"The following is a single-choice question on {subject}. Answer the question by replying A, B, C or D.",
                            "input": f"Question: {row[0]}\n{choices}\nAnswer: ",
                            "output": "",
                            "target": row[5],
                        }

                        dataset[split][subject]["data"].append(data_sample)

        return dataset
