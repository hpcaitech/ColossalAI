import argparse
import os

import openai
from colossal_eval.evaluate.evaluator import Evaluator
from colossal_eval.utils import jload


def main(args):
    assert len(args.answer_file_list) == len(
        args.model_name_list
    ), "The number of answer files and model names should be equal!"

    # load config
    config = jload(args.config_file)

    if config["language"] in ["cn", "en"]:
        # get metric settings for all categories
        metrics_per_category = {}
        for category in config["category"].keys():
            metrics_all = {}
            for metric_type, metrics in config["category"][category].items():
                metrics_all[metric_type] = metrics
            metrics_per_category[category] = metrics_all

        battle_prompt = None
        if args.battle_prompt_file:
            battle_prompt = jload(args.battle_prompt_file)

        gpt_evaluation_prompt = None
        if args.gpt_evaluation_prompt_file:
            gpt_evaluation_prompt = jload(args.gpt_evaluation_prompt_file)

        if len(args.model_name_list) == 2 and not battle_prompt:
            raise Exception("No prompt file for battle provided. Please specify the prompt file for battle!")

        if len(args.model_name_list) == 1 and not gpt_evaluation_prompt:
            raise Exception(
                "No prompt file for gpt evaluation provided. Please specify the prompt file for gpt evaluation!"
            )

        if args.gpt_model == "text-davinci-003" and args.gpt_with_reference:
            raise Exception(
                "GPT evaluation with reference is not supported for text-davinci-003. You should specify chat models such as gpt-3.5-turbo or gpt-4."
            )

        # initialize evaluator
        evaluator = Evaluator(
            metrics_per_category,
            battle_prompt,
            gpt_evaluation_prompt,
            args.gpt_model,
            config["language"],
            args.gpt_with_reference,
        )
        if len(args.model_name_list) == 2:
            answers_1 = jload(args.answer_file_list[0])
            answers_2 = jload(args.answer_file_list[1])

            answers1 = []
            for category, value in answers_1.items():
                answers1.extend(value["data"])

            answers2 = []
            for category, value in answers_2.items():
                answers2.extend(value["data"])

            assert len(answers1) == len(answers2), "The number of answers for two models should be equal!"

            evaluator.battle(answers1=answers1, answers2=answers2)
            evaluator.save(args.save_path, args.model_name_list)
        elif len(args.model_name_list) == 1:
            targets = jload(args.target_file)
            answers = jload(args.answer_file_list[0])

            references = []
            for category, value in targets["test"].items():
                references.extend(value["data"])

            predictions = []
            for category, value in answers.items():
                predictions.extend(value["data"])

            assert len(references) == len(
                predictions
            ), "The number of target answers and model answers should be equal!"

            evaluator.evaluate(
                answers=predictions, targets=references, save_path=args.save_path, model_name=args.model_name_list[0]
            )
            evaluator.save(args.save_path, args.model_name_list)
        else:
            raise ValueError("Unsupported number of answer files and model names!")
    else:
        raise ValueError(f'Unsupported language {config["language"]}!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColossalAI LLM evaluation pipeline.")
    parser.add_argument(
        "--config_file", type=str, default=None, required=True, help="path to the file of target results"
    )
    parser.add_argument("--battle_prompt_file", type=str, default=None, help="path to the prompt file for battle")
    parser.add_argument(
        "--gpt_evaluation_prompt_file", type=str, default=None, help="path to the prompt file for gpt evaluation"
    )
    parser.add_argument("--target_file", type=str, default=None, help="path to the target answer (ground truth) file")
    parser.add_argument(
        "--answer_file_list",
        type=str,
        nargs="+",
        default=[],
        required=True,
        help="path to the answer files of at most 2 models",
    )
    parser.add_argument(
        "--model_name_list", type=str, nargs="+", default=[], required=True, help="the names of at most 2 models"
    )
    parser.add_argument(
        "--gpt_model",
        default="gpt-3.5-turbo-16k",
        choices=["text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
        help="which GPT model to use for evaluation",
    )
    parser.add_argument(
        "--gpt_with_reference",
        default=False,
        action="store_true",
        help="whether to include reference answer in gpt evaluation",
    )
    parser.add_argument("--save_path", type=str, default="results", help="path to save evaluation results")
    parser.add_argument("--openai_key", type=str, default=None, required=True, help="Your openai key")
    args = parser.parse_args()

    if args.openai_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main(args)
