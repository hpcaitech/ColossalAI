import argparse
import copy
import os
from typing import Dict, List

import torch
import torch.distributed as dist
from colossal_eval import dataset, models, utils

import colossalai
from colossalai.logging import get_dist_logger

logger = get_dist_logger()


def rm_and_merge(world_size: int, save_path: str, model_names: List[str], dataset_names: Dict[str, List]) -> None:
    """
    Remove inference result per rank and merge them into one file.

    Args:
        world_size: Number of processes for inference.
        save_path: The folder for storing inference results.
        model_names: Names of models for inference.
        dataset_names: Names of dataset for inference.

    """

    for model_name in model_names:
        for dataset_name, categories in dataset_names.items():
            all_answers = {}
            for category in categories:
                all_answers[category] = {"data": []}
                answers = {"data": []}

                for r in range(world_size):
                    directory = os.path.join(
                        save_path, model_name, f"{dataset_name}_{category}_inference_results_rank{r}.json"
                    )
                    if not os.path.exists(directory):
                        raise Exception(
                            f"Directory {directory} not found. There may be an error during inference time."
                        )
                    else:
                        rank_answers = utils.jload(directory)
                        answers["data"].extend(rank_answers["data"])
                        answers["inference_kwargs"] = rank_answers["inference_kwargs"]

                for r in range(world_size):
                    try:
                        directory = os.path.join(
                            save_path, model_name, f"{dataset_name}_{category}_inference_results_rank{r}.json"
                        )
                        os.remove(directory)
                    except Exception as e:
                        print(e)

                all_answers[category] = answers

            logger.info(f"Save inference results of model {model_name} on dataset {dataset_name}.")
            utils.jdump(all_answers, os.path.join(save_path, model_name, f"{dataset_name}_inference_results.json"))

        logger.info(f"Save inference results of model {model_name} for all dataset.")
    logger.info(f"Save inference results of all models for all dataset.")


def main(args):
    colossalai.launch_from_torch(config={}, seed=42)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    inference_data = {}
    debug_args = {}
    few_shot_args = {}

    config = utils.jload(args.config)

    model_parameters = config["model"]
    dataset_parameters = config["dataset"]

    for dataset_parameter in dataset_parameters:
        path = dataset_parameter["path"]
        save_path = dataset_parameter["save_path"]
        dataset_name = dataset_parameter["name"]
        debug_args[dataset_name] = dataset_parameter["debug"]
        few_shot_args[dataset_name] = dataset_parameter["few_shot"]

        if not args.load_dataset:
            if os.path.exists(save_path):
                dataset_ = utils.jload(save_path)
                inference_data[dataset_name] = dataset_["test"]
            else:
                raise Exception(
                    "Can't find the converted dataset. You may set load_dataset True to store the dataset first."
                )

            continue

        dataset_class = eval(f"dataset.{dataset_parameter['dataset_class']}")
        if not issubclass(dataset_class, dataset.BaseDataset):
            raise ValueError(f"Dataset class {dataset_parameter['dataset_class']} is not a subclass of BaseDataset.")

        dataset_ = dataset_class(path, logger, dataset_parameter["few_shot"])

        dataset_.save(save_path)
        inference_data[dataset_name] = dataset_.dataset["test"]

    for model_parameter in model_parameters:
        model_name = model_parameter["name"]
        model_class = eval(f"models.{model_parameter['model_class']}")
        paramerters = model_parameter["parameters"]
        paramerters.update({"logger": logger})
        paramerters.update({"prompt_template": utils.prompt_templates[paramerters["prompt_template"]]})

        model_ = model_class(**paramerters)
        if not issubclass(model_class, models.BaseModel):
            raise ValueError(f"Model class {model_parameter['model_class']} is not a subclass of BaseModel.")

        for dataset_name, split_data in inference_data.items():
            start = 0
            for category, category_data in split_data.items():
                if few_shot_args[dataset_name] and category_data["inference_kwargs"].get("few_shot_data", None) is None:
                    raise Exception(f"Dataset {dataset_name} doesn't have few-shot data for category {category}!")

                answers_to_dump = copy.deepcopy(category_data)
                partition_size = len(category_data["data"]) // world_size
                redundant = len(category_data["data"]) % world_size

                # Ensure that the amount of data for inference is as consistent as possible across different processes.
                lengths = [partition_size for _ in range(world_size)]
                for j in range(redundant):
                    lengths[(j + start) % world_size] += 1

                start = (start + redundant) % world_size

                questions = category_data["data"][sum(lengths[0:rank]) : sum(lengths[0:rank]) + lengths[rank]]

                answers_per_rank = model_.inference(
                    questions, inference_kwargs=category_data["inference_kwargs"], debug=debug_args[dataset_name]
                )

                answers_to_dump["data"] = answers_per_rank

                utils.jdump(
                    answers_to_dump,
                    os.path.join(
                        args.inference_save_path,
                        model_name,
                        f"{dataset_name}_{category}_inference_results_rank{rank}.json",
                    ),
                )

        logger.info(f"Rank {rank} peak CUDA mem: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB")

        del model_
        torch.cuda.empty_cache()

    dist.barrier()
    if rank == 0:
        model_names = [model_parameter["name"] for model_parameter in model_parameters]
        dataset_names = {key: list(inference_data[key].keys()) for key in inference_data}
        rm_and_merge(world_size, args.inference_save_path, model_names, dataset_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColossalEval inference process.")
    parser.add_argument("--config", type=str, default=None, required=True, help="path to config file")
    parser.add_argument("--load_dataset", default=False, action="store_true")
    parser.add_argument("--inference_save_path", type=str, default=None, help="path to save inference results")
    args = parser.parse_args()

    main(args)
