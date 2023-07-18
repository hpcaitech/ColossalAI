import concurrent.futures
import os
import re
import time
from copy import deepcopy
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns
import tqdm
from utils import jdump, jload

ref_step_template = {
    "en":
        "Now please compare the answer with the {adjective} answer, determine whether the answer is able to achieve the same level of {metric}.\n\n",
    "cn":
        "请比较答案与上面的{adjective}答案，确定答案是否可以达到与该{adjective}答案同样水平的{metric}。\n\n"
}

ref_answer_template_general = {
    "en": "\nAn example answer with good quality is as follows:\n\n{answer}\n\n",
    "cn": "\n一个优质的示例答案如下：\n\n{answer}\n\n"
}

ref_answer_template_correctness = {
    "en": "\nA correct answer is as follows:\n\n{answer}\n\n",
    "cn": "\n标准答案如下：\n\n{answer}\n\n"
}


def get_battle_result(sys_prompt: str, user_prompt: str, id: int, max_tokens: int = 2048) -> Dict[str, Any]:
    """
    Get battle evaluation from GPT-4.

    Args:
        sys_prompt: prompt for the system.
        user_prompt: prompt for the user.
        id: id of the answers for comparison.
        max_tokens: the maximum number of tokens to generate in the chat completion.

    Returns:
        An evaluation of one comparison.
    """

    MAX_API_RETRY = 3
    for _ in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            evaluation = response["choices"][0]["message"]["content"]
            return {"evaluation": evaluation, "id": id}
        except Exception as e:
            print(e)
            time.sleep(1)
    print(f"Evaluation {id} failed after {MAX_API_RETRY} retries.")
    return {"evaluation": "", "id": id}


def parse_battle_score(evaluation: str) -> List[float]:
    """
    Parse evaluation from GPT-4 and get the scores of model 1 and 2.

    Args:
        evaluation: evaluation from GPT-4.

    Returns:
        A score pair of two different model answers.
    """

    try:
        pattern = re.compile("([0-9]|10) out of 10")
        sp = re.findall(pattern, evaluation)
        if len(re.findall(pattern, evaluation)) == 2:
            return [float(sp[0]), float(sp[1])]

        pattern = re.compile("a score of ([0-9]|10)")
        sp = re.findall(pattern, evaluation)
        if len(re.findall(pattern, evaluation)) == 2:
            return [float(sp[0]), float(sp[1])]

        pattern = re.compile("([0-9]|10)/10")
        sp = re.findall(pattern, evaluation)
        if len(re.findall(pattern, evaluation)) == 2:
            return [float(sp[0]), float(sp[1])]

        score_pair = evaluation.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception(f"Invalid score pair. Got {evaluation}.")
    except Exception as e:
        return [-1, -1]


def battle(answer1: List[Dict], answer2: List[Dict], prompt_dict: Dict[str, Any]) -> List[Dict]:
    """
    Use GPT-4 to compare answers of two different models.

    Args:
        answer1: answers of model 1.
        answer2: answers of model 2.
        prompt_dict: prompt for battle.

    Returns:
        Evaluations of all comparison pairs.
    """

    assert len(answer1) == len(answer2)

    handles = []
    evaluation_file = []

    total_len = len(answer1)
    question_idx_list = list(range(total_len))

    print(f" Total number of answers: {len(answer1)}.")

    evaluations = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in question_idx_list:
            assert answer1[i]["id"] == answer2[i]["id"]
            answer_id = answer1[i]["id"]

            ques = answer1[i]["instruction"] if answer1[i][
                "input"] == "" else answer1[i]["instruction"] + " " + answer1[i]["input"]
            cat = answer1[i]["category"]
            ans1 = answer1[i]["output"]
            ans2 = answer2[i]["output"]

            sys_prompt = prompt_dict["system_prompt"]
            prompt_template = prompt_dict["prompt_template"]
            prompt = prompt_template.format(
                question=ques,
                answer_1=ans1,
                answer_2=ans2,
                prompt=prompt_dict["prompt"],
            )

            future = executor.submit(get_battle_result, sys_prompt, prompt, answer_id, 2048)
            futures.append(future)

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            evaluations.append(future.result())

    evaluations.sort(key=lambda x: x["id"])

    return evaluations


def save_battle_results(evaluations: List[Dict], name1: str, name2: str, save_path: str) -> None:
    """
    Save evaluation results (model 1 vs model 2) from GPT-4.

    Args:
        evaluations: evaluation results from GPT-4.
        name1: model 1 's name.
        name2: model 2 's name.
        save_path: path to save battle results.
    """

    evaluation_file = deepcopy(evaluations)

    ans1_score = 0
    ans2_score = 0
    better_count = 0
    worse_count = 0
    tie_count = 0
    invalid_count = 0

    better_file = []
    worse_file = []
    tie_file = []
    invalid_file = []

    for idx, evaluation in enumerate(evaluations):
        scores = parse_battle_score(evaluation["evaluation"])
        evaluation_file[idx]["score"] = scores

        if scores[0] == -1 and scores[1] == -1:
            invalid_count += 1
            invalid_file.append(evaluation_file[idx])
            print(f'Invalid score pair: {evaluation_file[idx]["id"]}.')
        else:
            if scores[0] > scores[1]:
                worse_count += 1
                worse_file.append(evaluation_file[idx])
            elif scores[0] < scores[1]:
                better_count += 1
                better_file.append(evaluation_file[idx])
            else:
                tie_count += 1
                tie_file.append(evaluation_file[idx])
            ans1_score += scores[0]
            ans2_score += scores[1]

    prefix = f"{name1}_vs_{name2}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    jdump(better_file, os.path.join(save_path, prefix, f"{name2}_better.json"))
    jdump(worse_file, os.path.join(save_path, prefix, f"{name2}_worse.json"))
    jdump(tie_file, os.path.join(save_path, prefix, f"{prefix}_tie.json"))
    jdump(invalid_file, os.path.join(save_path, prefix, f"{prefix}_invalid.json"))
    jdump(evaluation_file, os.path.join(save_path, prefix, f"{prefix}_evaluations.json"))

    if os.path.exists(os.path.join(save_path, "battle_results.json")):
        results = jload(os.path.join(save_path, "battle_results.json"))
    else:
        results = {}

    results[prefix] = {
        "model": [name1, name2],
        "better": better_count,
        "worse": worse_count,
        "tie": tie_count,
        "win_rate": better_count / (len(evaluations) - invalid_count),
        "score": [
            ans1_score / (len(evaluations) - invalid_count),
            ans2_score / (len(evaluations) - invalid_count),
        ],
    }
    jdump(results, os.path.join(save_path, "battle_results.json"))

    print(f"Total {invalid_count} invalid score pair(s).")
    print(f"Model {name2} has {better_count} better answer(s).")
    print(f"Model {name2} has {worse_count} worse answer(s).")
    print(f"{tie_count} answer(s) play(s) to a tie.")
    print(f"Win rate of model {name2}: {better_count/(len(evaluations)-invalid_count):.2f}")
    print(f"Model {name1} average score: {ans1_score/(len(evaluations)-invalid_count):.2f}")
    print(f"Model {name2} average score: {ans2_score/(len(evaluations)-invalid_count):.2f}")


def reference_template(metric: str, language: str, reference: Dict[str, Any]) -> str:
    """
    Get prompt template for GPT evaluation with reference.

    Different languages have different prompt templates.

    Args:
        metric: metric used in GPT evaluation with reference.
        language: language for the template.
        reference: the instruction that contains target answer.

    Returns:
        Prompt template for GPT evaluation with reference.
    """

    step_to_add = ref_step_template[language]

    for_the_given_answer = "{metric} (1-5) (directly give the score for the given answer):" if language == "en" else "{metric} (1-5) (直接对给定答案打分)"

    # adjective is used to describe the word "answer" in the prompt.
    adjective = "example" if language == "en" else "示例"
    answer_to_add = ref_answer_template_general[language]

    # Only for correctness, we will provide a correct answer and so the adjective for "answer" will be "correct". The prompt words will be "a correct answer".
    # In other cases, the prompt words will be "an example answer with good quality" by default.
    if metric.lower() == "correctness":
        adjective = "correct" if language == "en" else "标准"
        answer_to_add = ref_answer_template_correctness[language]

    answer_to_add = answer_to_add.format(answer=reference["target"] if reference["target"] else reference["output"])
    step_to_add = step_to_add.format(metric=metric.lower(),
                                     adjective=adjective) + for_the_given_answer.format(metric=metric)

    return answer_to_add + step_to_add


def fill_in_message(role: str, content: str) -> Dict[str, str]:
    """
    Generate one formatted message to send through chat completion.

    Args:
        role: the role of the author of this message.
        content: the contents of the message.

    Returns:
        One message to send through chat completion.
    """

    return {"role": role, "content": content}


def multiturn_chat_completion(user_messages: List[str], model: str, max_tokens: int = 1, turns=2) -> Dict[str, Any]:
    """
    Do multi-turn chat completion.

    When turns == 1, it is a one-turn conversation for normal GPT evaluation.
    When turns == 2, it is a two-turn conversation which is used for GPT evaluation with reference answers.

    Args:
        user_messages: messages user wants to send.
        model: the model used to evaluate answers.
        max_tokens: the maximum number of tokens to generate in the chat completion.
        turns: the number of turns for conversation.

    Returns:
        Last turn's response.
    """

    if len(user_messages) != turns:
        raise Exception("The length of user messages should be equal to the turn number!")

    assistant_responses = []

    for i in range(turns):
        messages_to_send = []

        for j in range(i):
            messages_to_send.append(fill_in_message("user", user_messages[j]))
            messages_to_send.append(
                fill_in_message("assistant", assistant_responses[j]["choices"][0]["message"]["content"]))

        # Length of user messages == Length of assistant messages + 1
        # Because we always expect the api to response
        messages_to_send.append(fill_in_message("user", user_messages[i]))

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages_to_send,
            temperature=0,
            max_tokens=max_tokens,
        )

        # Avoid exceeding rate limits.
        # You can comment this line if your request doesn't contain many tokens.
        time.sleep(1)

        assistant_responses.append(response)

    return assistant_responses[-1]


def get_gpt_evaluation_without_logprobs(prompt: Dict[str, Any],
                                        inst: Dict[str, Any],
                                        metrics: List[str],
                                        language: str,
                                        reference: Dict[str, Any] = None,
                                        model: str = "gpt-3.5-turbo",
                                        max_tokens: int = 2048) -> Dict[str, Any]:
    """
    Use chat models(gpt-3.5-turbo or gpt-4) to evaluate one model answer.

    Temperature is set to 0 to make the model more deterministic.

    Args:
        prompt: a dictionary including prompt template, CoT and metrics.
        inst: the instruction that is needed to be evaluated.
        metrics: the metrics for evaluation.
        language: language used to change the CoT(add one more step about comparing the given answer and reference) if reference is not None.
        reference: the reference answer.
        model: the model used to evaluate answers.
        max_tokens: the maximum number of tokens to generate in the chat completion.

    Returns:
        An evaluation of one answer.
    """

    MAX_API_RETRY = 3

    question = (inst["instruction"] if inst["input"] == "" else inst["instruction"] + "\n" + inst["input"])
    answer = inst["output"]
    inst["evaluation"] = {}

    for metric in metrics:
        if prompt["metrics"].get(metric, None) is None:
            raise Exception(
                f"Unsupported metric {metric} for category {inst['category']}! You should add this metric in the prompt file!"
            )
        for i in range(MAX_API_RETRY):
            try:
                prompt_reference = "" if reference is None else reference_template(metric, language, reference)

                prompt_1st_round = prompt["prompt"].format(
                    question=question,
                    answer=answer,
                    metric=prompt["metrics"][metric],
                    steps=prompt["CoT"][metric],
                )

                if prompt_reference:
                    # Do a 2-round conversation
                    response = multiturn_chat_completion([prompt_1st_round, prompt_reference],
                                                         model,
                                                         max_tokens=max_tokens,
                                                         turns=2)
                else:
                    response = multiturn_chat_completion([prompt_1st_round], model, max_tokens=max_tokens, turns=1)

                inst["evaluation"][metric] = {
                    "response": response["choices"][0]["message"]["content"],
                    "logprobs": None,
                }

                # Prevent exceeding rate limits because we have multiple workers.
                # But this will slow down the evaluation process.
                # You can comment this line if your request doesn't contain many tokens.
                time.sleep(len(metrics) * 0.5)

                break
            except Exception as e:
                print(e)
                time.sleep(1)
        if metric not in inst["evaluation"]:
            print(f"Evaluation {inst['id']} for metric {metric} failed after {MAX_API_RETRY} retries.")
            inst["evaluation"][metric] = {}
    return inst


def get_gpt_evaluation_with_logprobs(prompt: Dict[str, Any],
                                     inst: Dict[str, Any],
                                     metrics: List[str],
                                     max_tokens: int = 2048) -> Dict[str, Any]:
    """
    Use completion model(text-davinci-003) to evaluate one model answer.
    Only completion models can return log probabilities.

    Temperature is set to 0 to make the model more deterministic.

    Args:
        prompt: a dictionary including prompt template, CoT and metrics.
        inst: the instruction that is needed to be evaluated.
        metrics: the metrics for evaluation.
        max_tokens: the maximum number of tokens to generate in the completion.

    Returns:
        An evaluation of one answer.
    """

    MAX_API_RETRY = 3

    question = (inst["instruction"] if inst["input"] == "" else inst["instruction"] + "\n" + inst["input"])
    answer = inst["output"]
    inst["evaluation"] = {}

    for metric in metrics:
        if prompt["metrics"].get(metric, None) is None:
            raise Exception(
                f"Unsupported metric {metric} for category {inst['category']}! You should add this metric in the prompt file!"
            )
        for i in range(MAX_API_RETRY):
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt["prompt"].format(
                        question=question,
                        answer=answer,
                        metric=prompt["metrics"][metric],
                        steps=prompt["CoT"][metric],
                    ),
                    logprobs=5,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                inst["evaluation"][metric] = {
                    "response": response["choices"][0]["text"],
                    "logprobs": response["choices"][0]["logprobs"]["top_logprobs"],
                }

                # Prevent exceeding rate limits because we have multiple workers.
                # But this will slow down the evaluation process.
                # You can comment this line if your request doesn't contain many tokens.
                time.sleep(len(metrics) * 0.5)

                break
            except Exception as e:
                print(e)
                time.sleep(1)
        if metric not in inst["evaluation"]:
            print(f"Evaluation {inst['id']} for metric {metric} failed after {MAX_API_RETRY} retries.")
            inst["evaluation"][metric] = {}
    return inst


def evaluate(answers: List[Dict],
             prompt: Dict[str, Any],
             metrics: List[str],
             category: str,
             model: str,
             language: str,
             references: List[Dict] = None) -> List[Dict]:
    """
    Use GPT models to evaluate model answers and save evaluation results.

    Args:
        answers: model answers.
        prompt: prompt for GPT evaluation.
        metrics: metrics for GPT evaluation.
        category: the category of the model answers for evaluation.
        model: the specific GPT model used to evaluate answers.
        language: language used in GPT evaluation
        references: references for GPT evaluation

    Returns:
        Evaluations of the given answers.
    """

    print(f"The number of instances of category {category}'s is {len(answers)}.")

    evaluations = []

    metrics_str = ", ".join(x for x in metrics)
    print(f"Category {category}'s metrics are {metrics_str}.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for idx, inst in enumerate(answers):
            # Completion models can return log probabilities.
            if model == "text-davinci-003":
                future = executor.submit(get_gpt_evaluation_with_logprobs, prompt, inst, metrics, 1)
            else:
                future = executor.submit(get_gpt_evaluation_without_logprobs,
                                         prompt,
                                         inst,
                                         metrics,
                                         language,
                                         reference=None if references is None else references[idx],
                                         model=model,
                                         max_tokens=1)

            futures.append(future)

        for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                desc=f"{category}: ",
                total=len(futures),
        ):
            evaluations.append(future.result())

    evaluations.sort(key=lambda x: x["id"])

    print(f"{category} done.")

    return evaluations


def calculate_scores_form_logprobs(logprobs: Dict[str, Any]) -> float:
    """
    Calculate the score according to log probabilities returned by text-davinci-003.

    Calculation formula:
        score = sum(score_i * exp(value)) where score_i is the score which corresponds to the key(predicted token) and value is its log probability.

    Ref: https://arxiv.org/abs/2303.16634
    This paper proposes NLG evaluation methods using text-davinci-003(log probabilities returned by completion models) and GPT-4(probabilities obtained by sampling).

    Args:
        logprobs: logprobs returned by openai.Completion.

    Returns:
        The score of one answer.
    """

    # GPT-3.5 only returns score of 1 to 5.
    prob = np.zeros(5)

    for key, value in logprobs.items():
        # Sometimes the key will be one byte of a unicode character which takes the form of "bytes:\\xe7".
        # It is meaningless and thus we don't calculate probability.
        if "bytes" in key:
            continue
        # results[0] is the score which corresponds to the key(predicted token).
        # For example, key "5" corresponds to score 5.
        results = re.findall(r"\d", key)
        if len(results) == 1:
            prob[int(results[0]) - 1] = prob[int(results[0]) - 1] + np.exp(value)

    score = np.dot(np.arange(1, 6), prob)

    return score


def calculate_scores_form_response(response: str, evaluation: Dict[str, Any]) -> int:
    """
    Calculate the score from the response returned by gpt-3.5-turbo or gpt-4.
    Different from text-davinci-003, this function directly calculates the score according to the plain response returned by gpt-3.5-turbo or gpt-4.
    Although text-davinci-003 can return log probabilities, it costs ten times as much as gpt-3.5-turbo.

    Args:
        response: logprobs returned by openai.Completion.
        evaluation: the evaluation corresponds to the question.

    Returns:
        The score of one answer.
    """

    try:
        results = re.findall(r"\d", response)
        if len(results) == 1:
            return int(results[0])
        else:
            raise Exception(f"Invalid score pair. Got {evaluation}.")
    except Exception as e:
        return 0


def save_gpt_evaluation_results(model_name: str, gpt_evaluation_results: Dict[str, Any],
                                save_path: str) -> Dict[str, Any]:
    """
    Save evaluation results for different categories for one model.

    Args:
        model_name: name of the model for saving evaluation results.
        gpt_evaluation_results: evaluations results for all of the model answers.
        save_path: path to save GPT evaluation statistics.
    """

    all_evaluations = []
    for category, evaluations in gpt_evaluation_results.items():
        jdump(evaluations, os.path.join(save_path, model_name, f"{category}_evaluation_results.json"))
        all_evaluations.extend(evaluations)

    jdump(all_evaluations, os.path.join(save_path, f"{model_name}_evaluation_results.json"))

    return all_evaluations


def save_gpt_evaluation_statistics(model_name: str, evaluations: List[Dict], save_path: str) -> None:
    """
    Generate statistics for one model.

    Args:
        model_name: name of the model for saving statistics.
        evaluations: evaluations for all of the model answers.
        save_path: path to save GPT evaluation statistics.
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_per_category = {}
    for evaluation in evaluations:
        category = evaluation["category"]
        if evaluation["category"] in data_per_category.keys():
            data_per_category[category].append(evaluation)
        else:
            data_per_category[category] = [evaluation]

    all_statistics = {}
    for category, data in data_per_category.items():
        metrics = data[0]["evaluation"].keys()
        scores = {metric: [] for metric in metrics}
        for evaluation in data:
            for metric in metrics:
                if evaluation["evaluation"][metric] == {}:
                    # This means after 3 retries, the server still returns an error and we set the score to 0.
                    scores[metric].append(0)
                elif evaluation["evaluation"][metric]["logprobs"] is not None:
                    scores[metric].append(
                        calculate_scores_form_logprobs(evaluation["evaluation"][metric]["logprobs"][0]))
                else:
                    scores[metric].append(
                        calculate_scores_form_response(evaluation["evaluation"][metric]["response"], evaluation))

        statistics = {}
        for metric in metrics:
            arg_sort = np.argsort(scores[metric])
            statistics[metric] = {}
            statistics[metric]["avg_score"] = sum(scores[metric]) / len(data)
            statistics[metric]["best_3"] = {data[i]["id"]: scores[metric][i] for i in arg_sort[-3:][::-1]}
            statistics[metric]["worst_3"] = {data[i]["id"]: scores[metric][i] for i in arg_sort[:3]}

        all_statistics[category] = statistics

    jdump(
        all_statistics,
        os.path.join(save_path, f"{model_name}_evaluation_statistics.json"),
    )


def analyze_gpt_evaluation_statistics(statistics_path: str, save_path: str) -> None:
    """
    Analyze and visualize all GPT evaluation statistics in the given directory.

    Args:
        statistics_path: path to all the models' statistics.
        save_path: path to save table and visualization results.
    """

    if not os.path.exists(statistics_path):
        raise Exception(f'The given directory "{statistics_path}" doesn\'t exist! No statistics found!')

    all_statistics = {}

    for file_name in os.listdir(statistics_path):
        if file_name.endswith("_evaluation_statistics.json"):
            model_name = file_name.split("_evaluation_statistics.json")[0]
            all_statistics[model_name] = jload(os.path.join(statistics_path, file_name))

    if len(list(all_statistics.keys())) == 0:
        raise Exception(f'There are no statistics in the given directory "{statistics_path}"!')

    frame_all = {
        "model": [],
        "category": [],
        "metric": [],
        "avg_score": [],
        "best_3": [],
        "worst_3": [],
    }
    frame_per_category = {}
    for model_name, model_statistics in all_statistics.items():
        for category, category_statistics in model_statistics.items():
            if frame_per_category.get(category) is None:
                frame_per_category[category] = {
                    "model": [],
                    "metric": [],
                    "avg_score": [],
                    "best_3": [],
                    "worst_3": [],
                }

            for metric, metric_statistics in category_statistics.items():
                frame_all["model"].append(model_name)
                frame_all["category"].append(category)
                frame_all["metric"].append(metric)
                frame_all["avg_score"].append(metric_statistics["avg_score"])
                frame_all["best_3"].append(metric_statistics["best_3"])
                frame_all["worst_3"].append(metric_statistics["worst_3"])

                frame_per_category[category]["model"].append(model_name)
                frame_per_category[category]["metric"].append(metric)
                frame_per_category[category]["avg_score"].append(metric_statistics["avg_score"])
                frame_per_category[category]["best_3"].append(metric_statistics["best_3"])
                frame_per_category[category]["worst_3"].append(metric_statistics["worst_3"])

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frame_all = pd.DataFrame(frame_all)
    frame_all.to_csv(os.path.join(save_path, "gpt_evaluation_statistics.csv"))

    for category in tqdm.tqdm(
            frame_per_category.keys(),
            desc=f"GPT evaluation: ",
            total=len(frame_per_category.keys()),
    ):
        data = pd.DataFrame(frame_per_category[category])

        sns.set()
        fig = plt.figure(figsize=(16, 10))
        plt.ylim((0, 5))

        fig = sns.barplot(x="metric", y="avg_score", hue="model", data=data, dodge=True)
        fig.set_title(f"Comparison between Different Models for Category {category.title()}")
        plt.xlabel("Evaluation Metric")
        plt.ylabel("Average Score")

        figure = fig.get_figure()
        figure.savefig(os.path.join(save_path, f"{category}.png"), dpi=400)

        plt.close()
