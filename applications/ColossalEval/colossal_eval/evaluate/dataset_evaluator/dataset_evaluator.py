from typing import Dict, List

import colossal_eval.evaluate.dataset_evaluator.metrics as metric_helper
import numpy as np
import tqdm

LabelBasedMetrics = ["first_token_accuracy", "matthews_correlation"]
LossBasedMetrics = ["perplexity", "ppl_score", "ppl_score_over_choices", "per_byte_perplexity", "per_byte_ppl_score"]
CombinedMetrics = ["combined_single_choice_accuracy"]
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
]


class DatasetEvaluator(object):
    """
    Dataset evaluator.

    """

    def __init__(self):
        pass

    def _calculate_label_metrics(self, metric: str, category: str):
        """Calculate label-based metrics."""
        weight = len(self.data[category]["data"]) / self.metric_total_length[metric]

        str_label_map = {
            choice: idx for idx, choice in enumerate(self.data[category]["inference_kwargs"]["all_classes"])
        }

        references = [str_label_map[sample["target"]] for sample in self.data[category]["data"]]
        [sample["output"] for sample in self.data[category]["data"]]

        flag = False
        softmaxs = []
        for i, sample in enumerate(self.data[category]["data"]):
            if np.any(np.isnan(np.array(list(sample["softmax_over_choices"].values())))):
                if not flag:
                    print(
                        f"NaN in the softmax, switch to exact match for category {category} in dataset {self.dataset_name} in model {self.model_name}."
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
                softmaxs.append(references[i] if score == 1 else -1)
            else:
                softmaxs.append(np.argmax(np.array(list(sample["softmax_over_choices"].values()))))

        references = np.array(references)
        softmaxs = np.array(softmaxs)
        scores = np.sum(references == softmaxs) / len(self.data[category]["data"]) * 100

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
        softmaxs = []
        for i, sample in enumerate(self.data[category]["data"]):
            if np.any(np.isnan(np.array(list(sample["softmax_over_choices"].values())))):
                if not flag:
                    print(
                        f"NaN in the softmax, switch to exact match for category {category} in dataset {self.dataset_name} in model {self.model_name}."
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
                softmaxs.append(references[i] if score == 1 else -1)
            else:
                softmaxs.append(np.argmax(np.array(list(sample["softmax_over_choices"].values()))))

        metric_method = eval("metric_helper." + metric)

        total_score = 0.0
        for prediction, reference, references_label, softmax in zip(
            predictions, references, references_labels, softmaxs
        ):
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

        references = [sample["target"] for sample in self.data[category]["data"]]
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
            elif metric in OtherMetrics:
                for category in self.suggested_categories[metric]:
                    self._calculate_other_metrics(metric, category)
                    pbar.update(1)

        return self.evaluation_results

    def get_evaluation_results(self, data: List[Dict], dataset_name: str, model_name: str, metrics: List[str]):
        """
        Evaluate inference data on the given metrics.

        Args:
            data: Data to be evaluated.
            dataset_name: Name of the dataset
            model_name: Name of the model
            metrics: Metrics used to evaluate.

        """
        self.data = data
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.categories = list(data.keys())
        self.metrics = metrics

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
            self.suggested_categories[metric] = metric_helper.metrics4subcategory[self.dataset_name][metric]
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
