import os
from typing import Dict, List, Union

import colossal_eval.evaluate.dataset_evaluator.metrics as metric_helper
import numpy as np
import tqdm
from colossal_eval.utils import jdump

import colossal_eval.evaluate.dataset_evaluator.gpt_judge as gpt_helper  # noqa

LabelBasedMetrics = ["first_token_accuracy", "matthews_correlation"]
LossBasedMetrics = [
    "perplexity",
    "ppl_score",
    "ppl_score_over_choices",
    "per_byte_perplexity",
    "per_byte_ppl_score",
    "loss_over_all_tokens",
]
CombinedMetrics = ["combined_single_choice_accuracy"]
GPTMetrics = ["mtbench_single_judge"]
OtherMetrics = [
    "f1_score",
    "f1_zh_score",
    "rouge_score",
    "rouge_zh_score",
    "retrieval_score",
    "retrieval_zh_score",
    "classification_score",
    "code_sim_score",
    "count_score",
    "multi_choice_accuracy",
    "math_equivalence",
    "single_choice_accuracy",
    "gsm_accuracy",
]


class DatasetEvaluator(object):
    """
    Dataset evaluator.

    """

    def __init__(self, config_path: str, save_path: str):
        self.config_path = config_path
        self.save_path = save_path

    def _calculate_label_metrics(self, metric: str, category: str):
        """Calculate label-based metrics."""
        weight = len(self.data[category]["data"]) / self.metric_total_length[metric]

        str_label_map = {
            choice: idx for idx, choice in enumerate(self.data[category]["inference_kwargs"]["all_classes"])
        }

        references = [str_label_map[sample["target"]] for sample in self.data[category]["data"]]
        [sample["output"] for sample in self.data[category]["data"]]

        flag = False
        logits = []
        for i, sample in enumerate(self.data[category]["data"]):
            if np.any(np.isnan(np.array(list(sample["logits_over_choices"].values())))):
                if not flag:
                    print(
                        f"NaN in the logits, switch to exact match for category {category} in dataset {self.dataset_name} in model {self.model_name}."
                    )
                    flag = True
                score = 0
                for ref in sample["target"]:
                    score = max(
                        score,
                        metric_helper.single_choice_accuracy(
                            sample["output"], ref, all_classes=self.data[category]["inference_kwargs"]["all_classes"]
                        ),
                    )

                    score = max(
                        score,
                        metric_helper.accuracy_by_options(sample["input"], sample["output"], ref),
                    )
                logits.append(references[i] if score == 1 else -1)
            else:
                logits.append(np.argmax(np.array(list(sample["logits_over_choices"].values()))))

        references = np.array(references)
        logits = np.array(logits)
        scores = np.sum(references == logits) / len(self.data[category]["data"]) * 100

        self.evaluation_results[metric][category] = (scores, len(self.data[category]["data"]))
        self.evaluation_results[metric]["ALL"] += scores * weight

    def _calculate_combined_metrics(self, metric: str, category: str):
        """Calculate combined metrics."""
        weight = len(self.data[category]["data"]) / self.metric_total_length[metric]

        references = [sample["target"] for sample in self.data[category]["data"]]
        predictions = [sample["output"] for sample in self.data[category]["data"]]

        str_label_map = {
            choice: idx for idx, choice in enumerate(self.data[category]["inference_kwargs"]["all_classes"])
        }

        references_labels = [str_label_map[sample["target"][0]] for sample in self.data[category]["data"]]
        predictions = [sample["output"] for sample in self.data[category]["data"]]

        flag = False
        logits = []
        for i, sample in enumerate(self.data[category]["data"]):
            if np.any(np.isnan(np.array(list(sample["logits_over_choices"].values())))):
                if not flag:
                    print(
                        f"NaN in the logits, switch to exact match for category {category} in dataset {self.dataset_name} in model {self.model_name}."
                    )
                    flag = True
                score = 0
                for ref in sample["target"]:
                    score = max(
                        score,
                        metric_helper.single_choice_accuracy(
                            sample["output"], ref, all_classes=self.data[category]["inference_kwargs"]["all_classes"]
                        ),
                    )
                logits.append(references[i] if score == 1 else -1)
            else:
                logits.append(np.argmax(np.array(list(sample["logits_over_choices"].values()))))

        metric_method = eval("metric_helper." + metric)

        total_score = 0.0
        for prediction, reference, references_label, softmax in zip(predictions, references, references_labels, logits):
            score = 0.0

            for ref in reference:
                score = max(
                    score,
                    metric_method(prediction, ref, all_classes=self.data[category]["inference_kwargs"]["all_classes"]),
                )
            if references_label == softmax:
                score = 1

            total_score += score
        total_score = total_score * 100 / len(self.data[category]["data"])

        self.evaluation_results[metric][category] = (total_score, len(self.data[category]["data"]))
        self.evaluation_results[metric]["ALL"] += total_score * weight

    def _calculate_other_metrics(self, metric: str, category: str):
        """Calculate other metrics."""
        weight = len(self.data[category]["data"]) / self.metric_total_length[metric]

        references = [
            sample["target"] if isinstance(sample["target"], list) else [sample["target"]]
            for sample in self.data[category]["data"]
        ]
        predictions = [sample["output"] for sample in self.data[category]["data"]]

        metric_method = eval("metric_helper." + metric)

        total_score = 0.0
        for prediction, reference in zip(predictions, references):
            score = 0.0
            for ref in reference:
                score = max(
                    score,
                    metric_method(prediction, ref, all_classes=self.data[category]["inference_kwargs"]["all_classes"]),
                )
            total_score += score
        total_score = total_score * 100 / len(predictions)

        self.evaluation_results[metric][category] = (total_score, len(self.data[category]["data"]))
        self.evaluation_results[metric]["ALL"] += total_score * weight

    def _calculate_gpt_metrics(self, metric: str, category: str):
        """Calculate gpt metrics."""
        weight = len(self.data[category]["data"]) / self.metric_total_length[metric]

        metric_method = eval("gpt_helper." + metric)

        judgements, avg_ratings = metric_method(self.data[category]["data"], self.config_path)
        self.judgements[category] = judgements

        self.evaluation_results[metric][category] = (np.mean(avg_ratings), len(self.data[category]["data"]))
        self.evaluation_results[metric]["ALL"] += np.mean(avg_ratings) * weight

        for i in range(avg_ratings.shape[0]):
            if f"{metric}_{i+1}" not in self.evaluation_results:
                self.evaluation_results[f"{metric}_{i+1}"] = {cat: 0 for cat in (["ALL"] + self.categories)}
            self.evaluation_results[f"{metric}_{i+1}"][category] = (avg_ratings[i], len(self.data[category]["data"]))
            self.evaluation_results[f"{metric}_{i+1}"]["ALL"] += avg_ratings[i] * weight

    def _calculate_loss_metrics(self, metric: str, category: str):
        """Calculate perplexity."""
        if metric == "perplexity":
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            losses = [min(sample["loss"]) for sample in self.data[category]["data"]]
            perplexity = np.mean(np.exp(np.array(losses)))

            self.evaluation_results["perplexity"][category] = (perplexity, len(self.data[category]["data"]))
            self.evaluation_results["perplexity"]["ALL"] += perplexity * weight
        elif metric == "ppl_score":
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            losses = [min(sample["loss"]) for sample in self.data[category]["data"]]
            perplexity_score = np.mean(np.exp(-np.array(losses))) * 100

            self.evaluation_results["ppl_score"][category] = (perplexity_score, len(self.data[category]["data"]))
            self.evaluation_results["ppl_score"]["ALL"] += perplexity_score * weight
        elif metric == "ppl_score_over_choices" and self.data[category]["inference_kwargs"]["all_classes"] is not None:
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            loss_over_choices = [sample["loss_over_choices"] for sample in self.data[category]["data"]]
            perplexity_score_over_choices = np.mean(np.exp(-np.array(loss_over_choices))) * 100

            self.evaluation_results["ppl_score_over_choices"][category] = (
                perplexity_score_over_choices,
                len(self.data[category]["data"]),
            )
            self.evaluation_results["ppl_score_over_choices"]["ALL"] += perplexity_score_over_choices * weight
        elif metric == "per_byte_perplexity":
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            losses = [min(sample["loss_sum"]) for sample in self.data[category]["data"]]
            perplexity = np.mean(np.exp(np.array(losses) / np.array(self.N_bytes[category])))

            self.evaluation_results["per_byte_perplexity"][category] = perplexity
            self.evaluation_results["per_byte_perplexity"]["ALL"] += perplexity * weight
        elif metric == "per_byte_ppl_score":
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            losses = [min(sample["loss_sum"]) for sample in self.data[category]["data"]]
            perplexity_score = np.mean(np.exp(-np.array(losses) / np.array(self.N_bytes[category]))) * 100

            self.evaluation_results["per_byte_ppl_score"][category] = perplexity_score
            self.evaluation_results["per_byte_ppl_score"]["ALL"] += perplexity_score * weight
        elif metric == "loss_over_all_tokens":
            weight = len(self.data[category]["data"]) / self.metric_total_length[metric]
            losses = [min(sample["loss_sum"]) for sample in self.data[category]["data"]]
            token_nums = [sample["token_num"][np.argmin(sample["loss_sum"])] for sample in self.data[category]["data"]]
            perplexity = np.sum(np.array(losses)) / np.sum(np.array(token_nums))

            self.evaluation_results["loss_over_all_tokens"][category] = perplexity
            self.evaluation_results["loss_over_all_tokens"]["ALL"] += perplexity * weight

            # The number of tokens can be used for normalizing.
            # See https://github.com/SkyworkAI/Skywork/issues/43#issuecomment-1811733834
            print(f"{self.model_name} {category} token num: {np.sum(np.array(token_nums))}")

    def _evaluate(self):
        """Calculate and return evaluation results"""

        for metric in self.metrics:
            pbar = tqdm.tqdm(
                desc=f"{self.dataset_name}-{metric}-{self.model_name}", total=len(self.suggested_categories[metric])
            )

            if metric in LabelBasedMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_label_metrics(metric, category)
                    pbar.update(1)
            elif metric in LossBasedMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_loss_metrics(metric, category)
                    pbar.update(1)
            elif metric in CombinedMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_combined_metrics(metric, category)
                    pbar.update(1)
            elif metric in GPTMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_gpt_metrics(metric, category)
                    pbar.update(1)
            elif metric in OtherMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_other_metrics(metric, category)
                    pbar.update(1)
            else:
                raise Exception(f"{metric} not supported.")

        if self.judgements:
            judgement_path = os.path.join(self.save_path, f"{self.model_name}_judgements.json")
            jdump(self.judgements, judgement_path)

        return self.evaluation_results

    def get_evaluation_results(
        self, data: Dict[str, Union[str, Dict]], dataset_name: str, model_name: str, metrics: List[str]
    ):
        """
        Evaluate inference data on the given metrics.

        Args:
            data: Data to be evaluated.
            dataset_name: Name of the dataset
            model_name: Name of the model
            metrics: Metrics used to evaluate.

        """
        self.data = data["inference_results"]
        self.dataset_name = dataset_name
        self.dataset_class = data["dataset_class"]
        self.model_name = model_name
        self.categories = list(self.data.keys())
        self.metrics = metrics
        self.judgements = {}

        self.evaluation_results = {
            metric: {category: 0 for category in (["ALL"] + self.categories)} for metric in self.metrics
        }

        self.total_length = 0
        self.total_single_choices = 0
        for value in self.data.values():
            self.total_length += len(value["data"])
            if value["inference_kwargs"]["all_classes"] is not None:
                self.total_single_choices += len(value["data"])

        self.metric_total_length = {metric: 0 for metric in self.metrics}
        self.suggested_categories = {metric: [] for metric in self.metrics}

        for metric in self.metrics:
            # Train and reference split use same metric as test split.
            self.suggested_categories[metric] = metric_helper.metrics4subcategory[self.dataset_class][metric]
            if "ALL" in self.suggested_categories[metric]:
                self.suggested_categories[metric] = self.categories
                self.metric_total_length[metric] = self.total_length
                continue
            for category in self.suggested_categories[metric]:
                self.metric_total_length[metric] += len(self.data[category]["data"])

        if "per_byte_perplexity" in self.metrics or "per_byte_ppl_score" in self.metrics:
            self.N_bytes = {category: [] for category in self.categories}
            for category in self.categories:
                samples = self.data[category]["data"]
                for sample in samples:
                    self.N_bytes[category].append(sample["byte_num"][0])

        return self._evaluate()
