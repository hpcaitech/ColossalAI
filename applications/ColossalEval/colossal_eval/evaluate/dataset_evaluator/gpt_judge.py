# Code adapted from https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge

import ast
import concurrent.futures
import copy
import json
import os
import re
import time
from typing import Any, Dict, List

import numpy as np
import openai
import tqdm

MODEL = "gpt-4"

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

NEED_REF_CATS = ["math", "reasoning", "coding"]

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def load_mt_prompts(prompt_file: str):
    prompts = {}
    with open(prompt_file) as fin:
        for line in fin:
            line = json.loads(line)
            prompts[line["name"]] = line
    return prompts


def get_mt_prompt(prompts: Dict[str, str], multiturn: bool, math: bool):
    if math and multiturn:
        return prompts["single-math-v1-multi-turn"]
    elif math and not multiturn:
        return prompts["single-math-v1"]
    elif not math and multiturn:
        return prompts["single-v1-multi-turn"]
    elif not math and not multiturn:
        return prompts["single-v1"]


def chat_compeletion_openai(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 2048):
    output = API_ERROR_OUTPUT
    model = MODEL
    for _ in range(API_MAX_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output


def get_mtbench_judgements(question: Dict[str, Any], prompts: Dict[str, str]):
    id = question["id"]
    judgement = {"id": id, "judgements": [], "ratings": []}
    category = question["category"]
    math = category in NEED_REF_CATS
    turn_number = len(question["instruction"])

    for num in range(turn_number):
        assert (len(question["target"]) >= 1 and math) or not math
        kwargs = {}
        if num >= 1:
            prompt = get_mt_prompt(prompts, multiturn=True, math=math)
            if len(question["target"]) >= 1 and math:
                kwargs = {f"ref_answer_{i+1}": question["target"][i] for i in range(len(question["target"]))}
            user_prompt = prompt["prompt_template"].format(
                question_1=question["instruction"][0],
                question_2=question["instruction"][1],
                answer_1=question["output"][0],
                answer_2=question["output"][1],
                **kwargs,
            )
        else:
            prompt = get_mt_prompt(prompts, multiturn=False, math=math)
            if len(question["target"]) >= 1 and math:
                kwargs = {"ref_answer_1": question["target"][0]}
            user_prompt = prompt["prompt_template"].format(
                question=question["instruction"][0],
                answer=question["output"][0],
                **kwargs,
            )

        rating = -1
        sys_prompt = prompt["system_prompt"]
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]

        judgement_str = chat_compeletion_openai(messages, temperature=0.0, max_tokens=2048)
        match = re.search(one_score_pattern, judgement_str)
        if not match:
            match = re.search(one_score_pattern_backup, judgement_str)
        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = -1

        judgement["judgements"].append(judgement_str)
        judgement["ratings"].append(rating)

    return judgement


def mtbench_single_judge(data: List[Dict], config_path: str):
    judgements = []

    prompt_dir = os.path.dirname(config_path)
    prompts = load_mt_prompts(os.path.join(prompt_dir, "mtbench_judge_prompts.jsonl"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, question in enumerate(data):
            future = executor.submit(get_mtbench_judgements, question, prompts)
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            desc=f"MTBench single judge for {data[0]['category']}",
            total=len(futures),
        ):
            judgements.append(future.result())

    judgements.sort(key=lambda x: x["id"])

    judgements_by_id = {j["id"]: j for j in judgements}

    data_to_dump = copy.deepcopy(data)

    for d in data_to_dump:
        id = d["id"]
        d["judgements"] = judgements_by_id[id]["judgements"]
        d["ratings"] = judgements_by_id[id]["ratings"]

    avg_ratings = np.mean([j["ratings"] for j in judgements], axis=0)

    return data_to_dump, avg_ratings
