import metrics as metrics
import pandas as pd


class Evaluator(object):
    """
        A class named Evaluator includes GPT-3.5/GPT-4 evaluation
        and automatic evaluation

    """

    def __init__(self, params: dict) -> None:
        self.params = params
        self.stats = dict()

    def battle(self, answers1: dict, answers2: dict) -> None:
        """
        Comparison between two models using GPT-4 as the reviewer.
        """
        pass

    def evaluate(self, answers: dict, targets: dict) -> None:
        """
        A comprehensive evaluation of the answers from the model.
        The function evaluates the model's performance from different perspectives
        using GPT-3.5, GPT-4, and off-the-shelf evaluation metrics.

        The metrics will be decided by the config file.

        """
        # automatic evaluation
        for category in self.params:
            category_metrics = self.params[category]["Metrics"]
            targets_list = []
            predicts_list = []
            self.stats[category] = {}

            for dict in targets:
                if dict["category"] == category:
                    if (dict["target"]):
                        targets_list.append(dict["target"])
                    else:
                        targets_list.append(dict["output"])

            for dict in answers:
                if dict["category"] == category:
                    predicts_list.append(dict["output"])

            for metric in category_metrics:
                if (metric == "BLEU"):
                    self.stats[category].update(metrics.bleu_score(preds=predicts_list, targets=targets_list))
                elif (metric == "ROUGE"):
                    self.stats[category].update(metrics.rouge_cn_score(preds=predicts_list, targets=targets_list))
                elif (metric == "Distinct"):
                    self.stats[category].update(metrics.distinct_score(preds=predicts_list))
                elif (metric == "BERTScore"):
                    self.stats[category].update(metrics.bert_score(preds=predicts_list, targets=targets_list))
                elif (metric == "Precision"):
                    self.stats[category].update(metrics.precision(preds=predicts_list, targets=targets_list))
                elif (metric == "Recall"):
                    self.stats[category].update(metrics.recall(preds=predicts_list, targets=targets_list))
                else:
                    self.stats[category].update(metrics.F1_score(preds=predicts_list, targets=targets_list))

    def save(self, path: str) -> None:
        # automatic evaluation result
        automatic_df = pd.DataFrame(self.stats)
        automatic_df.to_csv(path, index=True)
