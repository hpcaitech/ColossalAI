import argparse
import os

import tabulate
from colossal_eval.evaluate.dataset_evaluator import DatasetEvaluator
from colossal_eval.utils import jdump, jload


def main(args):
    config = jload(args.config)

    evaluation_results = {dataset["name"]: {} for dataset in config["dataset"]}
    evaluation_results_table = {dataset["name"]: {} for dataset in config["dataset"]}
    evaluator = DatasetEvaluator(args.config, args.evaluation_results_save_path)

    for dataset_parameter in config["dataset"]:
        dataset_name = dataset_parameter["name"]
        metrics = dataset_parameter["metrics"]
        results_metric_model = {metric: {model["name"]: None for model in config["model"]} for metric in metrics}
        for model in config["model"]:
            model_name = model["name"]

            data = jload(
                os.path.join(args.inference_results_path, model_name, f"{dataset_name}_inference_results.json")
            )
            results = evaluator.get_evaluation_results(data, dataset_name, model_name, metrics)

            for metric, score in results.items():
                if metric not in results_metric_model:
                    results_metric_model[metric] = {model["name"]: None for model in config["model"]}
                results_metric_model[metric][model_name] = score["ALL"]

            evaluation_results[dataset_name][model_name] = results

        evaluation_results_table[dataset_name] = results_metric_model

    table = []
    header = ["dataset", "metric"] + [model["name"] for model in config["model"]]
    table.append(header)

    for dataset_parameter in config["dataset"]:
        dataset_name = dataset_parameter["name"]
        metrics = dataset_parameter["metrics"]

        for metric, model_results in evaluation_results_table[dataset_name].items():
            row = [dataset_name]
            for model, score in model_results.items():
                if len(row) == 1:
                    row.extend([metric, "{:.02f}".format(score)])
                else:
                    row.append("{:.02f}".format(score))

            table.append(row)

    table = tabulate.tabulate(table, headers="firstrow")
    print(table)

    os.makedirs(args.evaluation_results_save_path, exist_ok=True)

    with open(os.path.join(args.evaluation_results_save_path, "evaluation_results_table.txt"), "w") as file:
        file.write(table)

    jdump(evaluation_results, os.path.join(args.evaluation_results_save_path, "evaluation_results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColossalEval evaluation process.")
    parser.add_argument("--config", type=str, default=None, required=True, help="path to config file")
    parser.add_argument("--inference_results_path", type=str, default=None, help="path to inference results")
    parser.add_argument(
        "--evaluation_results_save_path", type=str, default=None, help="path to save evaluation results"
    )
    args = parser.parse_args()

    main(args)
