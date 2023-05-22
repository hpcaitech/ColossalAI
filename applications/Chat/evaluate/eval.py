import argparse
import json

from evaluator import Evaluator
from utils import jload

def main(args):
     # load config
    config=jload(args.config_file)

    if config["language"] == "cn":
        # get metric settings for all categories
        metrics_per_category = {}
        for category in config["category"].keys():
            metrics_all = {}
            for metric_type, metrics in config["category"][category].items():
                metrics_all[metric_type]=metrics
            metrics_per_category[category]=metrics_all

        targets = jload(args.target_file)
        answers = jload(args.predict_file)
            
        # initialize evaluator
        evaluator = Evaluator(metrics_per_category)
        evaluator.evaluate(answers = answers, targets = targets)
        evaluator.save(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None, help='path to the file of target results')
    parser.add_argument('--target_file', type=str, default=None, help='path to the file of target results')
    parser.add_argument('--predict_file', type=str, default=None, help='path to the file of model prediction results')
    parser.add_argument('--save_path', type=str, default="results.csv", help='path to the csv file to save evaluation results')
    args = parser.parse_args()
    main(args)