import pandas as pd
from typing import Dict, List

import metrics as metrics

class Evaluator(object):
    """
        A class named Evaluator includes GPT-3.5/GPT-4 evaluation 
        and automatic evaluation
   
    """
    def __init__(self, params: dict) -> None:
        self.params = params
        self.stats = dict()
    
    def battle(self, answers1: List[Dict], answers2: List[Dict]) -> None:
        """
        Comparison between two models using GPT-4 as the reviewer.
        """
        pass

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

        # automatic evaluation
        for category in self.params:
            category_metrics = self.params[category]["Metrics"]
            targets_list = []
            predicts_list = []
            self.stats[category] = {}
            
            for dict in targets:
                if dict["category"] == category:
                    if(dict["target"]):
                        targets_list.append(dict["target"])
                    else:
                        targets_list.append(dict["output"])

            for dict in answers: 
                if dict["category"] == category:
                    predicts_list.append(dict["output"])

            for metric in category_metrics:
                self.stats[category].update(switch(metric=metric))

    def save(self, path: str) -> None:
        # automatic evaluation result
        automatic_df = pd.DataFrame(self.stats)
        automatic_df.to_csv(path,index=True)
