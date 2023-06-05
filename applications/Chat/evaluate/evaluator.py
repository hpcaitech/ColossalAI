import os
from typing import Any, Dict, List

import gpt_evaluate
import metrics
import pandas as pd
from utils import analyze_automatic_results, get_data_per_category, save_automatic_results


class Evaluator(object):
    """
        A class named Evaluator includes GPT-3.5/GPT-4 evaluation
        and automatic evaluation

    """

    def __init__(self, params: Dict[str, Any], battle_prompt: Dict[str, Any], gpt_evaluation_prompt: Dict[str, Any],
                 gpt_model: str, language: str) -> None:
        self.params = params
        self.battle_prompt = battle_prompt
        self.gpt_evaluation_prompt = gpt_evaluation_prompt
        self.gpt_model = gpt_model
        self.language = language
        self.automatic_metric_stats = dict()
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
            elif (metric == "Distinct"):
                return metrics.distinct_score(preds=predicts_list, language=language)
            elif (metric == "BERTScore"):
                return metrics.bert_score(preds=predicts_list, targets=targets_list, language=language)
            elif (metric == "Precision"):
                return metrics.precision(preds=predicts_list, targets=targets_list, language=language)
            elif (metric == "Recall"):
                return metrics.recall(preds=predicts_list, targets=targets_list, language=language)
            elif (metric == "F1 score"):
                return metrics.F1_score(preds=predicts_list, targets=targets_list, language=language)
            else:
                raise ValueError(f"Unexpected metric")

        answers_per_category = get_data_per_category(answers, list(self.params.keys()))
        targets_per_category = get_data_per_category(targets, list(self.params.keys()))

        # automatic evaluation
        for category in self.params:
            if len(answers_per_category[category]) == 0:
                print(f"Category {category} specified in your config doesn't have corresponding answers!")
                continue

            category_metrics = self.params[category]["Metrics"]
            self.automatic_metric_stats[category] = {}

            targets_list = [
                target["target"] if target["target"] else target["output"] for target in targets_per_category[category]
            ]
            predicts_list = [answer["output"] for answer in answers_per_category[category]]

            for metric in category_metrics:
                self.automatic_metric_stats[category].update(switch(metric=metric, language=self.language))

        # gpt evaluation
        for category in self.params:
            if len(answers_per_category[category]) == 0:
                print(f"Category {category} specified in your config doesn't have corresponding answers!")
                continue

            category_metrics = self.params[category]["GPT"]

            prompt = self.gpt_evaluation_prompt.get(category, None)
            if prompt is None:
                print(f"No prompt for category {category}! Use prompt for category general now.")
                prompt = self.gpt_evaluation_prompt["general"]

            self.gpt_evaluation_results[category] = gpt_evaluate.evaluate(answers_per_category[category], prompt,
                                                                          category_metrics, category, self.gpt_model)

    def save(self, path: str, model_name_list: List[str]) -> None:
        """
        Save evaluation results of GPT-3.5, GPT-4, and off-the-shelf evaluation metrics.

        """

        if len(model_name_list) == 2:
            save_path = os.path.join(path, "gpt_evaluate", "battle_results")
            gpt_evaluate.save_battle_results(self.battle_results, model_name_list[0], model_name_list[1], save_path)
        else:
            # Save evaluation results for automatic metrics
            automatic_base_save_path = os.path.join(path, "automatic_results")
            automatic_results_save_path = os.path.join(automatic_base_save_path, "evaluation_results")

            save_automatic_results(model_name_list[0], self.automatic_metric_stats, automatic_results_save_path)

            # Save charts and csv.
            automatic_analyses_save_path = os.path.join(automatic_base_save_path, "evaluation_analyses")
            analyze_automatic_results(automatic_results_save_path, automatic_analyses_save_path)

            # Save evaluation results for GPT evaluation metrics.
            gpt_base_save_path = os.path.join(path, "gpt_evaluate", "gpt_evaluate_results")
            gpt_evaluation_results_save_path = os.path.join(gpt_base_save_path, "evaluation_results")

            all_evaluations = gpt_evaluate.save_gpt_evaluation_results(model_name_list[0], self.gpt_evaluation_results,
                                                                       gpt_evaluation_results_save_path)

            # Start to calculate scores and save statistics.
            gpt_evaluation_statistics_save_path = os.path.join(gpt_base_save_path, "evaluation_statistics")
            gpt_evaluate.save_gpt_evaluation_statistics(model_name_list[0], all_evaluations,
                                                        gpt_evaluation_statistics_save_path)

            # Save charts and csv.
            gpt_evaluation_analyses_save_path = os.path.join(gpt_base_save_path, "evaluation_analyses")
            gpt_evaluate.analyze_gpt_evaluation_statistics(gpt_evaluation_statistics_save_path,
                                                           gpt_evaluation_analyses_save_path)
