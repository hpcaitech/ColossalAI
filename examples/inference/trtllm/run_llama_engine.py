import argparse

import tensorrt_llm

from colossalai.inference.trtllm.model.llama.builder_and_runner import LlamaEngineRunner, LlamaRunnerConfig

if __name__ == "__main__":
    args = LlamaRunnerConfig.add_args_argument(LlamaRunnerConfig, argparse.ArgumentParser()).parse_args()
    engine_runner = LlamaEngineRunner()
    runtime_rank = tensorrt_llm.mpi_rank()
    args.rank = runtime_rank
    outputs = engine_runner.generate(**vars(args))
    if runtime_rank == 0:
        print("outputs: ", outputs)
