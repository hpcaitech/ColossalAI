import os
from typing import Any, Dict, List

import gpt_evaluate
import metrics
import pandas as pd
import unieval
from utils import analyze_automatic_results, get_data_per_category, save_automatic_results


class Evaluator(object):
    """
        A class named Evaluator includes GPT-3.5/GPT-4 evaluation
        and automatic evaluation

    """

    def __init__(self, params: Dict[str, Any], battle_prompt: Dict[str, Any], gpt_evaluation_prompt: Dict[str, Any],
                 gpt_model: str, language: str, path_for_UniEval: Dict[str, str], gpt_with_reference: bool) -> None:
        self.params = params
        self.battle_prompt = battle_prompt
        self.gpt_evaluation_prompt = gpt_evaluation_prompt
        self.gpt_model = gpt_model
        self.language = language
        self.path_for_UniEval = path_for_UniEval
        self.gpt_with_reference = gpt_with_reference
        self.automatic_metric_stats = dict()
        self.unieval_metric_stats = dict()
        self.gpt_evaluation_results = dict()
        self.battle_results = []

    def battle(self, answers1: List[Dict], answers2: List[Dict]) -> None:
        """
        Comparison between two models using GPT-4 as the reviewer.
        """

        self.battle_results = gpt_evaluate.battle(answers1, answers2, self.battle_prompt)

    def evaluate(self, answers: List[Dict], targets: List[Dict]) -> None:
        """
        A comprehensive evaluation of the answers from the model.
        The function evaluates the model's performance from different perspectives
        using GPT-3.5, GPT-4, and off-the-shelf evaluation metrics.

        The metrics will be decided by the config file.

        """

        def switch(metric, language):
            if metric == "BLEU":
                return metrics.bleu_score(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "ROUGE":
                return metrics.rouge_score(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "Distinct":
                return metrics.distinct_score(preds=predicts_list, language=language)
            elif metric == "BERTScore":
                return metrics.bert_score(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "Precision":
                return metrics.precision(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "Recall":
                return metrics.recall(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "F1 score":
                return metrics.F1_score(preds=predicts_list, targets=targets_list, language=language)
            elif metric == "CHRF":
                return metrics.chrf_score(preds=predicts_list, targets=targets_list, language=language)
            else:
                raise ValueError(f"Unexpected metric")

        answers_per_category = get_data_per_category(answers, list(self.params.keys()))
        targets_per_category = get_data_per_category(targets, list(self.params.keys()))

        # automatic evaluation
        for category in self.params:
            if len(answers_per_category[category]) == 0:
                print(f"Category {category} specified in your config doesn't have corresponding answers!")
                continue

            if self.params[category].get("Metrics", None) is None:
                continue

            category_metrics = self.params[category]["Metrics"]
            self.automatic_metric_stats[category] = {}

            targets_list = [
                target["target"] if target["target"] else target["output"] for target in targets_per_category[category]
            ]
            predicts_list = [answer["output"] for answer in answers_per_category[category]]

            for metric in category_metrics:
                self.automatic_metric_stats[category].update(switch(metric=metric, language=self.language))

        # UniEval evaluation
        # self.unieval_metric_stats's key is "task" instead of "category".
        # Iterating "task" first will avoid repeated loading models because one task corresponds to one UniEval model.
        # If key is "category", different models will be loaded for multiple times across categories because the user may require different task(models) to evaluate one category.
        for category in self.params:
            if len(answers_per_category[category]) == 0:
                print(f"Category {category} specified in your config doesn't have corresponding answers!")
                continue

            if self.params[category].get("UniEval", None) is None:
                continue

            if self.params[category]["UniEval"] and self.language == "cn":
                raise Exception(
                    "UniEval doesn't support Chinese! Please remove UniEval config in your Chinese config file.")

            category_metrics = self.params[category]["UniEval"]

            for task, metric in [tuple(category_metric.split("-")) for category_metric in category_metrics]:
                if self.unieval_metric_stats.get(task, None) is None:
                    self.unieval_metric_stats[task] = {category: {metric: 0}}
                elif self.unieval_metric_stats[task].get(category, None) is None:
                    self.unieval_metric_stats[task][category] = {metric: 0}
                else:
                    self.unieval_metric_stats[task][category][metric] = 0

        for task in self.unieval_metric_stats:
            if self.path_for_UniEval is None:
                raise Exception(f"Please specify the path for UniEval model in the config file!")

            if self.path_for_UniEval.get(task, None) is None:
                raise Exception(f"Please specify the model path for task {task} in the config file!")

            print(f"Load UniEval model for task {task}.")

            uni_evaluator = unieval.get_evaluator(task, model_name_or_path=self.path_for_UniEval[task])
            for category in self.unieval_metric_stats[task]:
                targets_list = [
                    target["target"] if target["target"] else target["output"]
                    for target in targets_per_category[category]
                ]
                predicts_list = [answer["output"] for answer in answers_per_category[category]]
                sources_list = [answer["instruction"] + answer["input"] for answer in answers_per_category[category]]

                data = unieval.convert_data_to_unieval_format(predicts_list, sources_list, targets_list)
                scores = uni_evaluator.evaluate(data,
                                                category,
                                                dims=list(self.unieval_metric_stats[task][category].keys()),
                                                overall=False)
                avg_scores = unieval.calculate_average_score(scores)

                self.unieval_metric_stats[task][category].update(avg_scores)

        # gpt evaluation
        for category in self.params:
            if len(answers_per_category[category]) == 0:
                print(f"Category {category} specified in your config doesn't have corresponding answers!")
                continue

            if self.params[category].get("GPT", None) is None:
                continue

            category_metrics = self.params[category]["GPT"]

            prompt = self.gpt_evaluation_prompt.get(category, None)
            if prompt is None:
                print(f"No prompt for category {category}! Use prompt for category general now.")
                prompt = self.gpt_evaluation_prompt["general"]

            self.gpt_evaluation_results[category] = gpt_evaluate.evaluate(
                answers_per_category[category],
                prompt,
                category_metrics,
                category,
                self.gpt_model,
                self.language,
                references=targets_per_category[category] if self.gpt_with_reference else None)

    def save(self, path: str, model_name_list: List[str]) -> None:
        """
        Save evaluation results of GPT-3.5, GPT-4, and off-the-shelf evaluation metrics.

        """

        if len(model_name_list) == 2:
            save_path = os.path.join(path, "gpt_evaluate", "battle_results")
            gpt_evaluate.save_battle_results(self.battle_results, model_name_list[0], model_name_list[1], save_path)
        else:
            if self.automatic_metric_stats:
                # Save evaluation results for automatic metrics
                automatic_base_save_path = os.path.join(path, "automatic_results")
                automatic_results_save_path = os.path.join(automatic_base_save_path, "evaluation_results")

                save_automatic_results(model_name_list[0], self.automatic_metric_stats, automatic_results_save_path)

                # Save charts and csv.
                automatic_analyses_save_path = os.path.join(automatic_base_save_path, "evaluation_analyses")
                analyze_automatic_results(automatic_results_save_path, automatic_analyses_save_path)

            if self.unieval_metric_stats:
                # Save evaluation results for UniEval metrics
                unieval_base_save_path = os.path.join(path, "unieval_results")
                unieval_results_save_path = os.path.join(unieval_base_save_path, "evaluation_results")

                unieval.save_unieval_results(model_name_list[0], self.unieval_metric_stats, unieval_results_save_path)

                # Save charts and csv.
                unieval_analyses_save_path = os.path.join(unieval_base_save_path, "evaluation_analyses")
                unieval.analyze_unieval_results(unieval_results_save_path, unieval_analyses_save_path)

            if self.gpt_evaluation_results:
                # Save evaluation results for GPT evaluation metrics.
                gpt_base_save_path = os.path.join(path, "gpt_evaluate", "gpt_evaluate_results")
                gpt_evaluation_results_save_path = os.path.join(gpt_base_save_path, "evaluation_results")

                all_evaluations = gpt_evaluate.save_gpt_evaluation_results(model_name_list[0],
                                                                           self.gpt_evaluation_results,
                                                                           gpt_evaluation_results_save_path)

                # Start to calculate scores and save statistics.
                gpt_evaluation_statistics_save_path = os.path.join(gpt_base_save_path, "evaluation_statistics")
                gpt_evaluate.save_gpt_evaluation_statistics(model_name_list[0], all_evaluations,
                                                            gpt_evaluation_statistics_save_path)

                # Save charts and csv.
                gpt_evaluation_analyses_save_path = os.path.join(gpt_base_save_path, "evaluation_analyses")
                gpt_evaluate.analyze_gpt_evaluation_statistics(gpt_evaluation_statistics_save_path,
                                                               gpt_evaluation_analyses_save_path)
