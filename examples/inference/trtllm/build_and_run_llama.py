import argparse

import tensorrt_llm
import torch
from tensorrt_llm.logger import logger

from colossalai.inference.trtllm.model.llama.builder_and_runner import (
    LlamaArgsConfig,
    LlamaEngineBuilder,
    LlamaEngineRunner,
    LlamaRunnerConfig,
)


class BuildAndRunArgs(LlamaArgsConfig, LlamaRunnerConfig):
    @staticmethod
    def add_self_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = LlamaArgsConfig.add_self_args(parser)
        parser = LlamaRunnerConfig.add_self_args(parser)
        return parser


def build_and_run(rank: int, args: BuildAndRunArgs, model: str):
    builder_config = LlamaArgsConfig.init_from_args(args)
    generate_config = LlamaRunnerConfig.init_from_args(args)
    builder = LlamaEngineBuilder()
    engine_runner = LlamaEngineRunner()
    builder.set_config(builder_config)
    byte_engine, runner_config = builder.build(rank, "llama")

    generate_config.byte_engine = byte_engine
    generate_config.runner_config = runner_config
    generate_config.rank = rank

    outputs = engine_runner.generate(**vars(generate_config))

    if rank == 0:
        print("outputs: ", outputs)


if __name__ == "__main__":
    args = BuildAndRunArgs.add_args_argument(BuildAndRunArgs, argparse.ArgumentParser()).parse_args()
    rank = tensorrt_llm.mpi_rank()
    if args.parallel_build and args.world_size > 1 and torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f"Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free."
        )
        build_and_run(rank, args, "llama")
    else:
        args.parallel_build = False
        logger.info("Serially build TensorRT engines.")
        build_and_run(rank, args, "llama")
