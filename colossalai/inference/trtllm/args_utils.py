import argparse
import dataclasses
from dataclasses import dataclass


@dataclass
class BuilderArgsConfig:
    world_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    model_dir: str = None
    dtype: str = "float16"
    timing_cache: str = "model.cache"
    log_level: str = "info"
    vocab_size: int = 32000
    n_layer: int = 32
    n_positions: int = 2048
    n_embd: int = 4096
    n_head: int = 32
    inter_size: int = None
    hidden_act: str = "silu"
    max_batch_size: int = 8
    max_input_len: int = 2048
    max_output_len: int = 512
    max_beam_width: int = 1
    use_gpt_attention_plugin: str = False
    use_gemm_plugin: str = False
    parallel_build: bool = False
    enable_context_fmha: bool = False
    enable_context_fmha_fp32_acc: bool = False
    gpus_per_node: int = 8
    builder_opt: int = None
    output_dir: str = "llama_outputs"
    remove_input_padding: bool = False
    use_smooth_quant: bool = False
    per_channel: bool = False
    per_token: bool = False
    int8_kv_cache: bool = False
    use_parallel_embedding: bool = False
    embedding_sharding_dim: int = 1
    enable_fp8: bool = False
    fp8_kv_cache: bool = False
    use_weight_only: bool = False
    weight_only_precision: str = "int8"
    use_inflight_batching: bool = False
    paged_kv_cache: bool = False
    tokens_per_block: int = 64
    max_num_tokens: int = None
    strongly_typed: bool = False
    use_custom_all_reduce: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--world_size", type=int, default=1)
        parser.add_argument("--tp_size", type=int, default=1)
        parser.add_argument("--pp_size", type=int, default=1)
        parser.add_argument("--model_dir", type=str, default=None)
        parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "bfloat16", "float16"])
        parser.add_argument(
            "--timing_cache",
            type=str,
            default="model.cache",
            help="The path of to read timing cache from, will be ignored if the file does not exist",
        )
        parser.add_argument("--log_level", type=str, default="info")
        parser.add_argument("--vocab_size", type=int, default=32000)
        parser.add_argument("--n_layer", type=int, default=32)
        parser.add_argument("--n_positions", type=int, default=2048)
        parser.add_argument("--n_embd", type=int, default=4096)
        parser.add_argument("--n_head", type=int, default=32)
        parser.add_argument("--ffn_dim_multiplier", type=float, default=1.0)
        parser.add_argument("--inter_size", type=int, default=None)
        parser.add_argument("--hidden_act", type=str, default="silu")
        parser.add_argument("--max_batch_size", type=int, default=8)
        parser.add_argument("--max_input_len", type=int, default=2048)
        parser.add_argument("--max_output_len", type=int, default=512)
        parser.add_argument("--max_beam_width", type=int, default=1)
        parser.add_argument(
            "--use_gpt_attention_plugin",
            nargs="?",
            const="float16",
            type=str,
            default=False,
            choices=["float16", "bfloat16", "float32"],
        )
        parser.add_argument(
            "--use_gemm_plugin",
            nargs="?",
            const="float16",
            type=str,
            default=False,
            choices=["float16", "bfloat16", "float32"],
        )
        parser.add_argument("--parallel_build", default=False, action="store_true")
        parser.add_argument("--enable_context_fmha", default=False, action="store_true")
        parser.add_argument("--enable_context_fmha_fp32_acc", default=False, action="store_true")
        parser.add_argument("--gpus_per_node", type=int, default=8)
        parser.add_argument("--builder_opt", type=int, default=None)
        parser.add_argument(
            "--output_dir",
            type=str,
            default="llama_outputs",
            help="The path to save the serialized engine files, timing cache file and model configs",
        )
        parser.add_argument("--remove_input_padding", default=False, action="store_true")

        # Arguments related to the quantization of the model.
        parser.add_argument(
            "--use_smooth_quant",
            default=False,
            action="store_true",
            help="Use the SmoothQuant method to quantize activations and weights for the various GEMMs."
            "See --per_channel and --per_token for finer-grained quantization options.",
        )
        parser.add_argument(
            "--per_channel",
            default=False,
            action="store_true",
            help="By default, we use a single static scaling factor for the GEMM's result. "
            "per_channel instead uses a different static scaling factor for each channel. "
            "The latter is usually more accurate, but a little slower.",
        )
        parser.add_argument(
            "--per_token",
            default=False,
            action="store_true",
            help="By default, we use a single static scaling factor to scale activations in the int8 range. "
            "per_token chooses at run time, and for each token, a custom scaling factor. "
            "The latter is usually more accurate, but a little slower.",
        )
        parser.add_argument(
            "--int8_kv_cache",
            default=False,
            action="store_true",
            help="By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV",
        )
        parser.add_argument(
            "--use_parallel_embedding",
            action="store_true",
            default=False,
            help="By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled",
        )
        parser.add_argument(
            "--embedding_sharding_dim",
            type=int,
            default=1,  # Meta does TP on hidden dim
            choices=[0, 1],
            help="By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). "
            "To shard it along hidden dimension, set embedding_sharding_dim=1"
            "Note: embedding sharing is only enabled when embedding_sharding_dim = 0",
        )
        parser.add_argument(
            "--enable_fp8",
            default=False,
            action="store_true",
            help="Use FP8 Linear layer for Attention QKV/Dense and MLP.",
        )
        parser.add_argument(
            "--fp8_kv_cache",
            default=False,
            action="store_true",
            help="By default, we use dtype for KV cache. fp8_kv_cache chooses int8 quantization for KV",
        )
        parser.add_argument(
            "--use_weight_only",
            default=False,
            action="store_true",
            help="Quantize weights for the various GEMMs to INT4/INT8."
            "See --weight_only_precision to set the precision",
        )
        parser.add_argument(
            "--weight_only_precision",
            const="int8",
            type=str,
            nargs="?",
            default="int8",
            choices=["int8", "int4", "int4_awq", "int4_gptq"],
            help="Define the precision for the weights when using weight-only quantization."
            "You must also use --use_weight_only for that argument to have an impact.",
        )
        parser.add_argument(
            "--use_inflight_batching",
            action="store_true",
            default=False,
            help="Activates inflight batching mode of gptAttentionPlugin.",
        )
        parser.add_argument(
            "--paged_kv_cache",
            action="store_true",
            default=False,
            help="By default we use contiguous KV cache. By setting this flag you enable paged KV cache",
        )
        parser.add_argument(
            "--tokens_per_block", type=int, default=64, help="Number of tokens per block in paged KV cache"
        )
        parser.add_argument(
            "--max_num_tokens", type=int, default=None, help="Define the max number of tokens supported by the engine"
        )
        parser.add_argument(
            "--strongly_typed",
            default=False,
            action="store_true",
            help="This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.",
        )
        parser.add_argument(
            "--use_custom_all_reduce",
            action="store_true",
            help="Activates latency-optimized algorithm for all-reduce instead of NCCL.",
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BuilderArgsConfig":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args


@dataclass
class RunnerArgsConfig:
    max_output_len: int
    log_level: str = "error"
    engine_dir: str = "llama_outputs"
    tokenizer_dir: str = "."
    input_text: str = "Born in north-east France, Soyer trained as a"
    input_tokens: str = None
    output_csv: str = None
    num_beams: int = 1
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 0.0
    debug_mode: bool = False
    streaming: bool = False
    streaming_interval: int = 5
    model_name: str = ""
    use_fast: bool = False
    trust_remote_code: bool = False
    encoder_max_input_length: int = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_output_len", type=int, required=True)
        parser.add_argument("--log_level", type=str, default="error")
        parser.add_argument("--engine_dir", type=str, default="llama_outputs")
        parser.add_argument("--tokenizer_dir", type=str, default=".", help="Directory containing the tokenizer.model.")
        parser.add_argument("--input_text", type=str, default="Born in north-east France, Soyer trained as a")
        parser.add_argument(
            "--input_tokens",
            dest="input_file",
            type=str,
            help="CSV or Numpy file containing tokenized input. Alternative to text input.",
            default=None,
        )
        parser.add_argument(
            "--output_csv", type=str, help="CSV file where the tokenized output is stored.", default=None
        )
        parser.add_argument(
            "--output_npy", type=str, help="Numpy file where the tokenized output is stored.", default=None
        )
        parser.add_argument("--num_beams", type=int, help="Use beam search if num_beams >1", default=1)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--top_k", type=int, default=1)
        parser.add_argument("--top_p", type=float, default=0.0)
        parser.add_argument("--debug_mode", default=False, action="store_true")
        parser.add_argument("--streaming", default=False, action="store_true")
        parser.add_argument(
            "--streaming_interval", type=int, help="How often to return tokens when streaming.", default=5
        )
        parser.add_argument("--model_name", type=str, default="")
        parser.add_argument(
            "--use_fast", default=False, help="Whether or not to use a fast tokenizer.", action="store_true"
        )
        parser.add_argument(
            "--trust_remote_code",
            default=False,
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
            action="store_true",
        )
        parser.add_argument(
            "--encoder_max_input_length",
            help="The maximum input length of TensorRT-LLM encoder.",
            type=int,
            default=None,
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "RunnerArgsConfig":
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args
