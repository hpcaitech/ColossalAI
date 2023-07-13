import io
import json
import os
import re
import string
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
from zhon import hanzi


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_json_list(file_path):
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


def get_data_per_category(data, categories):
    data_per_category = {category: [] for category in categories}
    for item in data:
        category = item["category"]
        if category in categories:
            data_per_category[category].append(item)

    return data_per_category


def remove_punctuations(text: str) -> str:
    """
    Remove punctuations in the given text.
    It is used in evaluation of automatic metrics.

    """

    punctuation = string.punctuation + hanzi.punctuation
    punctuation = set([char for char in punctuation])
    punctuation.difference_update(set("!@#$%&()<>?|,.\"'"))

    out = []
    for char in text:
        if char in punctuation:
            continue
        else:
            out.append(char)

    return "".join(out)


def remove_redundant_space(text: str) -> str:
    """
    Remove redundant spaces in the given text.
    It is used in evaluation of automatic metrics.

    """

    return " ".join(text.split())


def preprocessing_text(text: str) -> str:
    """
    Preprocess the given text.
    It is used in evaluation of automatic metrics.

    """

    return remove_redundant_space(remove_punctuations(text.lower()))


def save_automatic_results(model_name: str, automatic_metric_stats: Dict[str, Dict], save_path: str) -> None:
    """
    Save automatic evaluation results of different categories for one model.

    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    automatic_df = pd.DataFrame(automatic_metric_stats)
    automatic_df.to_csv(os.path.join(save_path, f"{model_name}_results.csv"), index=True)


def read_automatic_results(results_path: str, file_name: str) -> Dict[str, Dict]:
    """
    Read a csv file and return a dictionary which stores scores per metric.

    """

    results = pd.read_csv(os.path.join(results_path, file_name), index_col=0)

    results_dict = {metric: {} for metric in list(results.index)}
    for i, metric in enumerate(results_dict.keys()):
        for j, category in enumerate(list(results.columns)):
            if pd.isnull(results.iloc[i][j]):
                continue
            results_dict[metric][category] = results.iloc[i][j]

    return results_dict


def analyze_automatic_results(results_path: str, save_path: str) -> None:
    """
    Analyze and visualize all csv files in the given folder.

    """

    if not os.path.exists(results_path):
        raise Exception(f'The given directory "{results_path}" doesn\'t exist! No results found!')

    all_statistics = {}

    for file_name in os.listdir(results_path):
        if file_name.endswith("_results.csv"):
            model_name = file_name.split("_results.csv")[0]
            all_statistics[model_name] = read_automatic_results(results_path, file_name)

    if len(list(all_statistics.keys())) == 0:
        raise Exception(f'There are no csv files in the given directory "{results_path}"!')

    frame_all = {"model": [], "category": [], "metric": [], "score": []}
    frame_per_metric = {}
    for model_name, model_statistics in all_statistics.items():
        for metric, metric_statistics in model_statistics.items():
            if frame_per_metric.get(metric) is None:
                frame_per_metric[metric] = {"model": [], "category": [], "score": []}

            for category, category_score in metric_statistics.items():
                frame_all["model"].append(model_name)
                frame_all["category"].append(category)
                frame_all["metric"].append(metric)
                frame_all["score"].append(category_score)

                frame_per_metric[metric]["model"].append(model_name)
                frame_per_metric[metric]["category"].append(category)
                frame_per_metric[metric]["score"].append(category_score)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    frame_all = pd.DataFrame(frame_all)
    frame_all.to_csv(os.path.join(save_path, "automatic_evaluation_statistics.csv"))

    for metric in tqdm.tqdm(
            frame_per_metric.keys(),
            desc=f"automatic metrics: ",
            total=len(frame_per_metric.keys()),
    ):
        data = pd.DataFrame(frame_per_metric[metric])

        sns.set()
        fig = plt.figure(figsize=(16, 10))

        fig = sns.barplot(x="category", y="score", hue="model", data=data, dodge=True)
        fig.set_title(f"Comparison between Different Models for Metric {metric.title()}")
        plt.xlabel("Evaluation Category")
        plt.ylabel("Score")

        figure = fig.get_figure()
        figure.savefig(os.path.join(save_path, f"{metric}.png"), dpi=400)

        plt.close()
