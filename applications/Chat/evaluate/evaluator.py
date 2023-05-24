import os
from typing import Any, Dict, List

import gpt_evaluate
import metrics
import pandas as pd
from utils import get_data_per_category, jdump


class Evaluator(object):
    """
        A class named Evaluator includes GPT-3.5/GPT-4 evaluation
        and automatic evaluation

    """

    def __init__(self, params: Dict[str, Any], battle_prompt: Dict[str, Any], gpt_evaluation_prompt: Dict[str,
                                                                                                          Any]) -> None:
        self.params = params
        self.battle_prompt = battle_prompt
        self.gpt_evaluation_prompt = gpt_evaluation_prompt
        self.automatic_metric_stats = dict()
        self.gpt35_evaluation_results = dict()
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

        def switch(metric):
            if metric == "BLEU":
                return metrics.bleu_score(preds=predicts_list, targets=targets_list)
            elif metric == "ROUGE":
                return metrics.rouge_cn_score(preds=predicts_list, targets=targets_list)
            elif (metric == "Distinct"):
                return metrics.distinct_score(preds=predicts_list)
            elif (metric == "BERTScore"):
                return metrics.bert_score(preds=predicts_list, targets=targets_list)
            elif (metric == "Precision"):
                return metrics.precision(preds=predicts_list, targets=targets_list)
            elif (metric == "Recall"):
                return metrics.recall(preds=predicts_list, targets=targets_list)
            elif (metric == "F1 score"):
                return metrics.F1_score(preds=predicts_list, targets=targets_list)
            else:
                raise ValueError(f"Unexpected metric")

        answers_per_category = get_data_per_category(answers, list(self.params.keys()))
        targets_per_category = get_data_per_category(targets, list(self.params.keys()))

        # automatic evaluation
        for category in self.params:
            category_metrics = self.params[category]["Metrics"]
            self.automatic_metric_stats[category] = {}

            targets_list = [
                target["target"] if target["target"] else target["output"] for target in targets_per_category[category]
            ]
            predicts_list = [answer["output"] for answer in answers_per_category[category]]

            for metric in category_metrics:
                self.automatic_metric_stats[category].update(switch(metric=metric))

        # gpt35 evaluation
        for category in self.params:
            category_metrics = self.params[category]["GPT-3.5"]

            prompt = self.gpt_evaluation_prompt.get(category, None)
            if prompt is None:
                print(f"No prompt for category {category}! Use prompt for category general now.")
                prompt = self.gpt_evaluation_prompt["general"]

            self.gpt35_evaluation_results[category] = gpt_evaluate.gpt35_evaluate(answers_per_category[category],
                                                                                  prompt, category_metrics, category)

    def save(self, path: str, model_name_list: List[str]) -> None:
        """
        Save evaluation results of GPT-3.5, GPT-4, and off-the-shelf evaluation metrics.

        """

        if len(model_name_list) == 2:
            save_path = os.path.join(path, "gpt_evaluate", "battle_results")
            gpt_evaluate.save_battle_results(self.battle_results, model_name_list[0], model_name_list[1], save_path)
        else:
            # save evaluation results for automatic metrics
            automatic_df = pd.DataFrame(self.automatic_metric_stats)

            automatic_results_save_path = os.path.join(path, "automatic_results")
            if not os.path.exists(automatic_results_save_path):
                os.makedirs(automatic_results_save_path)
            automatic_df.to_csv(os.path.join(automatic_results_save_path, f"{model_name_list[0]}.csv"), index=True)

            # Save evaluation results for GPT-3.5 evaluation metrics.
            all_evaluations = []
            base_save_path = os.path.join(path, "gpt_evaluate", "gpt35_evaluate_results")
            evaluation_results_save_path = os.path.join(base_save_path, "evaluation_results")

            for category, evaluations in self.gpt35_evaluation_results.items():
                jdump(
                    evaluations,
                    os.path.join(evaluation_results_save_path, model_name_list[0],
                                 f"{category}_evaluation_results.json"))
                all_evaluations.extend(evaluations)

            jdump(all_evaluations,
                  os.path.join(evaluation_results_save_path, f"{model_name_list[0]}_evaluation_results.json"))

            # Start to calculate scores and save statictics.
            evaluation_statistics_save_path = os.path.join(base_save_path, "evaluation_statistics")
            gpt_evaluate.save_gpt35_evaluation_statistics(model_name_list[0], all_evaluations,
                                                          evaluation_statistics_save_path)

            # Save charts and csv.
            evaluation_analyses_save_path = os.path.join(base_save_path, "evaluation_analyses")
            gpt_evaluate.analyze_gpt35_evaluation_statistics(evaluation_statistics_save_path,
                                                             evaluation_analyses_save_path)
