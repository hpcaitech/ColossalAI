# Adapted from https://github.com/ruixiangcui/AGIEval/blob/main/src/dataset_loader.py.

import ast
import glob
import os
from copy import deepcopy
from typing import Dict, List

import pandas as pd
from colossal_eval.utils import get_json_list

from colossalai.logging import DistributedLogger

from .base import BaseDataset

# define the datasets
english_qa_datasets = [
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "sat-math",
    "sat-en",
    "aqua-rat",
    "sat-en-without-passage",
    "gaokao-english",
]
chinese_qa_datasets = [
    "logiqa-zh",
    "jec-qa-kd",
    "jec-qa-ca",
    "gaokao-chinese",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-physics",
    "gaokao-mathqa",
]
english_cloze_datasets = ["math"]
chinese_cloze_datasets = ["gaokao-mathcloze"]

multi_choice_datasets = ["jec-qa-kd", "jec-qa-ca", "gaokao-physics", "gaokao-mathqa"]
math_output_datasets = {"gaokao-mathcloze", "math"}

default_inference_kwargs = {
    "calculate_loss": True,
    "all_classes": None,
    "language": "Chinese",
    "pretrain": False,
    "max_new_tokens": 32,
}


def get_prompt(line: Dict, dataset_name: str, logger: DistributedLogger) -> Dict:
    """Modified from https://github.com/microsoft/AGIEval/blob/main/src/dataset_loader.py#L190"""
    try:
        all_classes = None
        passage = line["passage"] if line["passage"] is not None else ""

        if dataset_name in english_qa_datasets:
            option_string = "ABCDEFG"
            count = len(line["options"])

            input = (
                "Question: "
                + line["question"]
                + " "
                + "Choose from the following options: "
                + " ".join(line["options"])
                + "\n"
                + "Answer: "
            )

            all_classes = list(option_string[0:count])

        elif dataset_name in chinese_qa_datasets:
            option_string = "ABCDEFG"
            count = len(line["options"])

            input = "问题：" + line["question"] + " " + "从以下选项中选择：" + " ".join(line["options"]) + "\n" + "答案："

            all_classes = list(option_string[0:count])

        elif dataset_name in english_cloze_datasets:
            input = "Question: " + line["question"] + "\n" + "Answer: "

        elif dataset_name in chinese_cloze_datasets:
            input = "问题：" + line["question"] + "\n" + "答案："

        return {
            "instruction": input if not passage else passage + "\n\n" + input,
            "target": line["label"] if line["label"] else line["answer"],
        }, all_classes

    except NameError:
        logger.info("Dataset not defined.")


# process few-shot raw_prompts
def combine_prompt(prompt_path, dataset_name, load_explanation=True, chat_mode=False):
    demostrations = []
    demostration_en = "Here are the answers for the problems in the exam."
    demostration_zh = "以下是考试中各个问题的答案。"

    if dataset_name in english_qa_datasets or dataset_name in english_cloze_datasets:
        demostrations.append(demostration_en)
    elif dataset_name in chinese_qa_datasets or dataset_name in chinese_cloze_datasets:
        demostrations.append(demostration_zh)

    skip_passage = False
    if dataset_name == "sat-en-without-passage":
        skip_passage = True
        dataset_name = "sat-en"

    # read the prompts by context and explanation
    context_row = [0, 1, 3, 5, 7, 9]
    explanation_row = [0, 2, 4, 6, 8, 10]
    raw_prompts_context = pd.read_csv(
        prompt_path, header=0, skiprows=lambda x: x not in context_row, keep_default_na=False
    )
    raw_prompts_explanation = pd.read_csv(
        prompt_path, header=0, skiprows=lambda x: x not in explanation_row, keep_default_na=False
    ).replace(r"\n\n", "\n", regex=True)
    contexts = []
    for line in list(raw_prompts_context[dataset_name]):
        if line:
            # print(line)
            contexts.append(ast.literal_eval(line))
    explanations = [exp for exp in raw_prompts_explanation[dataset_name] if exp]

    for idx, (con, exp) in enumerate(zip(contexts, explanations)):
        passage = con["passage"] if con["passage"] is not None and not skip_passage else ""
        question = con["question"]
        options = con["options"] if con["options"] is not None else ""
        label = con["label"] if con["label"] is not None else ""
        answer = con["answer"] if "answer" in con and con["answer"] is not None else ""

        if dataset_name in english_qa_datasets:
            question_input = (
                "Question: "
                + passage
                + " "
                + question
                + "\n"
                + "Choose from the following options: "
                + " ".join(options)
                + "\n"
                + "Answer: {}".format(label)
            )
        elif dataset_name in chinese_qa_datasets:
            question_input = (
                "问题：" + passage + " " + question + "\n" + "从以下选项中选择：" + " ".join(options) + "\n" + "答案：{}".format(label)
            )
        elif dataset_name in english_cloze_datasets:
            question_input = "Question: ".format(idx + 1) + question + "\n" + "Answer: {}".format(answer)
        elif dataset_name in chinese_cloze_datasets:
            question_input = "问题：" + question + "\n" + "答案：{}".format(answer)
        else:
            raise ValueError(f"During loading few-sot examples, found unknown dataset: {dataset_name}")

        if chat_mode:
            demostrations.append((question_input,))
        else:
            demostrations.append(question_input)

    return demostrations


class AGIEvalDataset(BaseDataset):
    """
    Dataset wrapper for AGIEval dataset.
    Data source: https://github.com/microsoft/AGIEval
    This dataset class will convert the original dataset into the inference dataset.

    A few dirty data needed to be manually corrected in the origin dataset:
    Issue link: https://github.com/microsoft/AGIEval/issues/16
    1. Invalid options in line 190 in gaokao-chemistry.jsonl.
    2. Option D (They may increase in value as those same resources become rare on Earth.) missing in line 17 in sat-en-without-passage.jsonl.
    3. Option D (They may increase in value as those same resources become rare on Earth.) missing in line 17 in sat-en.jsonl.
    4. Option D (No, because the data do not indicate whether the honeybees had been infected with mites.) missing in line 57 in sat-en-without-passage.jsonl.
    5. Option D (No, because the data do not indicate whether the honeybees had been infected with mites.) missing in line 57 in sat-en.jsonl.
    6. Option D (Published theories of scientists who developed earlier models of the Venus flytrap) missing in line 98 in sat-en-without-passage.jsonl.
    7. Option D (Published theories of scientists who developed earlier models of the Venus flytrap) missing in line 98 in sat-en.jsonl.
    8. Label is empty in line 212 in jec-qa-kd.jsonl. Content is also dirty.
    9. Actually, gaokao-mathqa.jsonl is also a multi-choice dataset. See line 149 286 287.
    """

    @staticmethod
    def load(
        path: str, logger: DistributedLogger, few_shot: bool, forward_only: bool, load_train: bool, load_reference: bool
    ) -> List[Dict]:
        dataset = {"test": {}}

        files = glob.glob(os.path.join(path, "*.jsonl"))
        files.sort()

        if few_shot:
            prompt_path = os.path.join(path, "few_shot_prompts.csv")

        for file in files:
            dataset_name = os.path.basename(file)[0 : -len(".jsonl")]

            few_shot_data = []
            if few_shot:
                # process demo once if it is few-shot-CoT
                few_shot_data = combine_prompt(prompt_path, dataset_name, load_explanation=False, chat_mode=False)

            dataset["test"][dataset_name] = {"data": []}

            file_dir = os.path.join(path, file)

            loaded_jsonl = get_json_list(file_dir)

            # It's been tested that each data sample in one subcategory have same inference arguments.
            _, all_classes = get_prompt(loaded_jsonl[0], dataset_name, logger)
            inference_kwargs = deepcopy(default_inference_kwargs)
            if all_classes is not None and dataset_name not in multi_choice_datasets:
                inference_kwargs["all_classes"] = all_classes

            if dataset_name in english_qa_datasets:
                inference_kwargs["language"] = "English"
            if dataset_name in chinese_qa_datasets:
                inference_kwargs["language"] = "Chinese"
            inference_kwargs["few_shot_data"] = few_shot_data

            dataset["test"][dataset_name]["inference_kwargs"] = inference_kwargs

            for line in loaded_jsonl:
                info, all_classes = get_prompt(line, dataset_name, logger)

                # Convert multi-choice answers to a single string.
                # We will convert it back when evaluating.
                # We do this because if target is a list, it should be only used for multiple target answers.
                if dataset_name in multi_choice_datasets:
                    if isinstance(info["target"], str) and len(info["target"]) > 1:
                        # "gaokao-mathqa" actually contain multi-choice questions.
                        # This if clause is specially used for it.
                        info["target"] = "".join(info["target"].split())
                    else:
                        info["target"] = "".join(info["target"])

                if isinstance(info["target"], list) and len(info["target"]) == 1:
                    info["target"] = info["target"][0]

                data_sample = {
                    "dataset": "agieval",
                    "split": "test",
                    "category": dataset_name,
                    "instruction": info["instruction"],
                    "input": "",
                    "output": "",
                    "target": info["target"],
                }

                dataset["test"][dataset_name]["data"].append(data_sample)

        return dataset
