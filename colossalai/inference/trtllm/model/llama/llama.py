import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.layers.attention import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import fp8_quantize, smooth_quantize, weight_only_groupwise_quantize, weight_only_quantize
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from transformers import LlamaConfig, LlamaForCausalLM

from colossalai.inference.trtllm.args_utils import BuilderArgsConfig, RunnerArgsConfig
from colossalai.inference.trtllm.engine_builder_base import EngineBuilderBase
from colossalai.inference.trtllm.engine_runner_base import EngineRunnerBase

from .weight import (
    get_scaling_factors,
    load_from_awq_llama,
    load_from_binary,
    load_from_gptq_llama,
    load_from_hf_llama,
    load_from_meta_llama,
    parse_ft_config,
)

MODEL_NAME = "llama"


@dataclass
class LlamaArgsConfig(BuilderArgsConfig):
    ft_model_dir: str = None
    meta_ckpt_dir: str = None
    quant_ckpt_path: str = None
    n_kv_head: int = None
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1.0
    rms_norm_eps: float = 1e-06
    rotary_base: float = 10000
    rotary_scaling: str = None
    use_rmsnorm_plugin: str = False
    visualize: bool = False
    enable_debug_output: bool = False
    per_group: bool = False
    group_size: int = 128
    quantized_fp8_model_path: str = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = BuilderArgsConfig.add_cli_args(parser)
        parser.add_argument("--ft_model_dir", type=str, default=None)
        parser.add_argument("--meta_ckpt_dir", type=str, default=None)
        parser.add_argument("--quant_ckpt_path", type=str, default=None)
        parser.add_argument("--n_kv_head", type=int, default=None)
        parser.add_argument("--multiple_of", type=int, default=256)
        parser.add_argument("--rms_norm_eps", type=float, default=1e-06)
        parser.add_argument("--rotary_base", type=float, default=10000.0)
        parser.add_argument("--rotary_scaling", nargs=2, type=str, default=None)
        parser.add_argument(
            "--use_rmsnorm_plugin",
            nargs="?",
            const="float16",
            type=str,
            default=False,
            choices=["float16", "float32", "bfloat16"],
        )
        parser.add_argument("--visualize", default=False, action="store_true")
        parser.add_argument("--enable_debug_output", default=False, action="store_true")
        parser.add_argument(
            "--per_group",
            default=False,
            action="store_true",
            help="By default, we use a single static scaling factor to scale weights in the int4 range. "
            "per_group chooses at run time, and for each group, a custom scaling factor. "
            "The flag is built for GPTQ/AWQ quantization.",
        )
        parser.add_argument("--group_size", type=int, default=128, help="Group size used in GPTQ/AWQ quantization.")
        parser.add_argument(
            "--quantized_fp8_model_path",
            type=str,
            default=None,
            help="Path of a quantized model checkpoint in .npz format",
        )

        return parser


@dataclass
class LlamaRunnerConfig(RunnerArgsConfig):
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = RunnerArgsConfig.add_cli_args(parser)
        return parser


class LlamaEngineBuilder(EngineBuilderBase):
    def __init__(self):
        super(LlamaEngineBuilder, self).__init__()

    def _process_config(self) -> None:
        assert (
            self._builder_args_config
        ), "builder_args_config is not initialized! Please initialize builder_args_config first."

        logger.set_level(self._builder_args_config.log_level)
        assert not (
            self._builder_args_config.use_smooth_quant and self._builder_args_config.use_weight_only
        ), "You cannot enable both SmoothQuant and INT8 weight-only together."

        if not self._builder_args_config.remove_input_padding:
            if self._builder_args_config.use_gpt_attention_plugin:
                logger.warning(f"It is recommended to specify --remove_input_padding when using GPT attention plugin")

        if self._builder_args_config.use_inflight_batching:
            if not self._builder_args_config.use_gpt_attention_plugin:
                self._builder_args_config.use_gpt_attention_plugin = "float16"
                logger.info(
                    f"Using GPT attention plugin for inflight batching mode. Setting to default '{args.use_gpt_attention_plugin}'"
                )
            if not self._builder_args_config.remove_input_padding:
                self._builder_args_config.remove_input_padding = True
                logger.info("Using remove input padding for inflight batching mode.")
            if not self._builder_args_config.paged_kv_cache:
                self._builder_args_config.paged_kv_cache = True
                logger.info("Using paged KV cache for inflight batching mode.")

        if self._builder_args_config.use_smooth_quant:
            self._builder_args_config.quant_mode = QuantMode.use_smooth_quant(
                self._builder_args_config.per_token, self._builder_args_config.per_channel
            )
        elif self._builder_args_config.use_weight_only:
            if self._builder_args_config.per_group:
                self._builder_args_config.quant_mode = QuantMode.from_description(
                    quantize_weights=True,
                    quantize_activations=False,
                    per_token=False,
                    per_channel=False,
                    per_group=True,
                    use_int4_weights=True,
                )
            else:
                self._builder_args_config.quant_mode = QuantMode.use_weight_only(
                    self._builder_args_config.weight_only_precision == "int4"
                )
        else:
            self._builder_args_config.quant_mode = QuantMode(0)

        if self._builder_args_config.int8_kv_cache:
            self._builder_args_config.quant_mode = self._builder_args_config.quant_mode.set_int8_kv_cache()
        elif self._builder_args_config.fp8_kv_cache:
            self._builder_args_config.quant_mode = self._builder_args_config.quant_mode.set_fp8_kv_cache()
        if self._builder_args_config.enable_fp8:
            self._builder_args_config.quant_mode = self._builder_args_config.quant_mode.set_fp8_qdq()

        if self._builder_args_config.rotary_scaling is not None:
            rotary_scaling = {
                "type": self._builder_args_config.rotary_scaling[0],
                "factor": float(self._builder_args_config.rotary_scaling[1]),
            }
            assert rotary_scaling["type"] in ["linear", "dynamic"]
            assert rotary_scaling["factor"] > 1.0
            self._builder_args_config.rotary_scaling = rotary_scaling
            if rotary_scaling["type"] == "dynamic":
                assert not self._builder_args_config.remove_input_padding, "TODO: Not supported yet"

        # Since gpt_attenttion_plugin is the only way to apply RoPE now,
        # force use the plugin for now with the correct data type.
        self._builder_args_config.use_gpt_attention_plugin = self._builder_args_config.dtype
        if self._builder_args_config.model_dir is not None:
            hf_config = LlamaConfig.from_pretrained(self._builder_args_config.model_dir)
            self._builder_args_config.inter_size = hf_config.intermediate_size  # override the inter_size for LLaMA
            self._builder_args_config.n_embd = hf_config.hidden_size
            self._builder_args_config.n_head = hf_config.num_attention_heads
            if hasattr(hf_config, "num_key_value_heads"):
                self._builder_args_config.n_kv_head = hf_config.num_key_value_heads
            self._builder_args_config.n_layer = hf_config.num_hidden_layers
            self._builder_args_config.n_positions = hf_config.max_position_embeddings
            self._builder_args_config.vocab_size = hf_config.vocab_size
            self._builder_args_config.hidden_act = hf_config.hidden_act
            self._builder_args_config.rms_norm_eps = hf_config.rms_norm_eps
        elif self._builder_args_config.meta_ckpt_dir is not None:
            with open(Path(self._builder_args_config.meta_ckpt_dir, "params.json")) as fp:
                meta_config: dict = json.load(fp)
            self._builder_args_config.n_embd = meta_config["dim"]
            self._builder_args_config.n_head = meta_config["n_heads"]
            self._builder_args_config.n_layer = meta_config["n_layers"]
            self._builder_args_config.n_kv_head = meta_config.get("n_kv_heads", self._builder_args_config.n_head)
            self._builder_args_config.multiple_of = meta_config["multiple_of"]
            self._builder_args_config.ffn_dim_multiplier = meta_config.get("ffn_dim_multiplier", 1)
            n_embd = int(4 * self._builder_args_config.n_embd * 2 / 3)
            self._builder_args_config.inter_size = self._builder_args_config.multiple_of * (
                (int(n_embd * self._builder_args_config.ffn_dim_multiplier) + self._builder_args_config.multiple_of - 1)
                // self._builder_args_config.multiple_of
            )
            self._builder_args_config.rms_norm_eps = meta_config["norm_eps"]
        elif self._builder_args_config.ft_model_dir is not None:
            n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head = parse_ft_config(
                Path(self._builder_args_config.ft_model_dir) / "config.ini"
            )
            self._builder_args_config.inter_size = inter_size  # override the inter_size for LLaMA
            self._builder_args_config.n_kv_head = n_kv_head
            self._builder_args_config.n_embd = n_embd
            self._builder_args_config.n_head = n_head
            self._builder_args_config.n_layer = n_layer
            self._builder_args_config.n_positions = n_positions
            self._builder_args_config.vocab_size = vocab_size
            self._builder_args_config.hidden_act = hidden_act
            self._builder_args_config.rms_norm_eps = 1e-06
            logger.warning("Set rms_norm_eps to 1e-06 directly.")
        assert self._builder_args_config.use_gpt_attention_plugin, "LLaMa must use gpt attention plugin"
        if self._builder_args_config.n_kv_head is None:
            self._builder_args_config.n_kv_head = self._builder_args_config.n_head
        elif self._builder_args_config.n_kv_head != self._builder_args_config.n_head:
            assert (
                self._builder_args_config.n_head % self._builder_args_config.n_kv_head
            ) == 0, "MQA/GQA requires the number of heads to be divisible by the number of K/V heads."
            assert (self._builder_args_config.n_kv_head % self._builder_args_config.tp_size) == 0 or (
                self._builder_args_config.tp_size % self._builder_args_config.n_kv_head
            ) == 0, (
                "MQA/GQA requires either the number of K/V heads to be divisible by the tensor parallelism size OR "
                "the tensor parallelism size to be divisible by the number of K/V heads."
            )

        if self._builder_args_config.dtype == "bfloat16":
            assert self._builder_args_config.use_gemm_plugin, "Please use gemm plugin when dtype is bfloat16"

        assert (
            self._builder_args_config.pp_size * self._builder_args_config.tp_size
            == self._builder_args_config.world_size
        )

        if self._builder_args_config.max_num_tokens is not None:
            assert self._builder_args_config.enable_context_fmha

        if self._builder_args_config.inter_size is None:
            # this should not be need when loading a real model
            # but it is helpful when creating a dummy model without loading any real weights
            n_embd = int(4 * self._builder_args_config.n_embd * 2 / 3)
            self._builder_args_config.inter_size = self._builder_args_config.multiple_of * (
                (int(n_embd * self._builder_args_config.ffn_dim_multiplier) + self._builder_args_config.multiple_of - 1)
                // self._builder_args_config.multiple_of
            )
            logger.info(f"Setting inter_size to {self._builder_args_config.inter_size}.")

    def _generate_network(self, rank: int) -> None:
        # Module -> Network
        self._network = self._builder.create_network()
        self._network.trt_network.name = self._engine_name
        if self._builder_args_config.use_gpt_attention_plugin:
            self._network.plugin_config.set_gpt_attention_plugin(
                dtype=self._builder_args_config.use_gpt_attention_plugin
            )
        if self._builder_args_config.use_gemm_plugin:
            self._network.plugin_config.set_gemm_plugin(dtype=self._builder_args_config.use_gemm_plugin)
        if self._builder_args_config.use_rmsnorm_plugin:
            self._network.plugin_config.set_rmsnorm_plugin(dtype=self._builder_args_config.use_rmsnorm_plugin)

        # Quantization plugins.
        if self._builder_args_config.use_smooth_quant:
            self._network.plugin_config.set_smooth_quant_gemm_plugin(dtype=self._builder_args_config.dtype)
            self._network.plugin_config.set_rmsnorm_quantization_plugin(dtype=self._builder_args_config.dtype)
            self._network.plugin_config.set_quantize_tensor_plugin()
            self._network.plugin_config.set_quantize_per_token_plugin()
        assert not (
            self._builder_args_config.enable_context_fmha and self._builder_args_config.enable_context_fmha_fp32_acc
        )
        if self._builder_args_config.enable_context_fmha:
            self._network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self._builder_args_config.enable_context_fmha_fp32_acc:
            self._network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
        if self._builder_args_config.use_weight_only:
            if self._builder_args_config.per_group:
                self._network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(dtype="float16")
            else:
                self._network.plugin_config.set_weight_only_quant_matmul_plugin(dtype="float16")
        if self._builder_args_config.world_size > 1:
            self._network.plugin_config.set_nccl_plugin(
                self._builder_args_config.dtype, self._builder_args_config.use_custom_all_reduce
            )
        if self._builder_args_config.remove_input_padding:
            self._network.plugin_config.enable_remove_input_padding()
        if self._builder_args_config.paged_kv_cache:
            self._network.plugin_config.enable_paged_kv_cache(self._builder_args_config.tokens_per_block)

    def _set_model(self, rank: int) -> None:
        dtype = str_dtype_to_trt(self._builder_args_config.dtype)
        mapping = Mapping(
            world_size=self._builder_args_config.world_size,
            rank=rank,
            tp_size=self._builder_args_config.tp_size,
            pp_size=self._builder_args_config.pp_size,
        )
        assert (
            self._builder_args_config.n_layer % self._builder_args_config.pp_size == 0
        ), f"num_layers {self._builder_args_config.n_layer} must be a multiple of pipeline parallelism size {self._builder_args_config.pp_size}"

        # Initialize Module
        self._trt_model = tensorrt_llm.models.LLaMAForCausalLM(
            num_layers=self._builder_args_config.n_layer,
            num_heads=self._builder_args_config.n_head,
            num_kv_heads=self._builder_args_config.n_kv_head,
            hidden_size=self._builder_args_config.n_embd,
            vocab_size=self._builder_args_config.vocab_size,
            hidden_act=self._builder_args_config.hidden_act,
            max_position_embeddings=self._builder_args_config.n_positions,
            dtype=dtype,
            mlp_hidden_size=self._builder_args_config.inter_size,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            mapping=mapping,
            rotary_base=self._builder_args_config.rotary_base,
            rotary_scaling=self._builder_args_config.rotary_scaling,
            use_parallel_embedding=self._builder_args_config.use_parallel_embedding,
            embedding_sharding_dim=self._builder_args_config.embedding_sharding_dim,
            quant_mode=self._builder_args_config.quant_mode,
            rms_norm_eps=self._builder_args_config.rms_norm_eps,
        )

        if self._builder_args_config.use_smooth_quant:
            self._trt_model = smooth_quantize(self._trt_model, self._builder_args_config.quant_mode)
        elif self._builder_args_config.use_weight_only:
            if self._builder_args_config.weight_only_precision == "int8":
                self._trt_model = weight_only_quantize(self._trt_model, self._builder_args_config.quant_mode)
            elif self._builder_args_config.weight_only_precision == "int4":
                self._trt_model = weight_only_quantize(self._trt_model, self._builder_args_config.quant_mode)
            elif self._builder_args_config.weight_only_precision == "int4_awq":
                self._trt_model = weight_only_groupwise_quantize(
                    model=self._trt_model,
                    quant_mode=self._builder_args_config.quant_mode,
                    group_size=self._builder_args_config.group_size,
                    zero=False,
                    pre_quant_scale=True,
                    exclude_modules=[],
                )
            elif self._builder_args_config.weight_only_precision == "int4_gptq":
                self._trt_model = weight_only_groupwise_quantize(
                    model=self._trt_model,
                    quant_mode=self._builder_args_config.quant_mode,
                    group_size=self._builder_args_config.group_size,
                    zero=True,
                    pre_quant_scale=False,
                )
        elif self._builder_args_config.enable_fp8 or self._builder_args_config.fp8_kv_cache:
            logger.info(f"Loading scaling factors from " f"{self._builder_args_config.quantized_fp8_model_path}")
            quant_scales = get_scaling_factors(
                self._builder_args_config.quantized_fp8_model_path,
                num_layers=self._builder_args_config.n_layer,
                quant_mode=self._builder_args_config.quant_mode,
            )
            self._trt_model = fp8_quantize(
                self._trt_model, quant_mode=self._builder_args_config.quant_mode, quant_scales=quant_scales
            )
        if self._builder_args_config.per_group:
            load_func = (
                load_from_awq_llama
                if self._builder_args_config.weight_only_precision == "int4_awq"
                else load_from_gptq_llama
            )
            load_func(
                tensorrt_llm_llama=self._trt_model,
                quant_ckpt_path=self._builder_args_config.quant_ckpt_path,
                mapping=mapping,
                dtype=self._builder_args_config.dtype,
            )
        elif self._builder_args_config.meta_ckpt_dir is not None:
            load_from_meta_llama(
                self._trt_model, self._builder_args_config.meta_ckpt_dir, mapping, self._builder_args_config.dtype
            )
        elif self._builder_args_config.model_dir is not None:
            logger.info(f"Loading HF LLaMA ... from {self._builder_args_config.model_dir}")
            tik = time.time()
            hf_llama = LlamaForCausalLM.from_pretrained(
                self._builder_args_config.model_dir,
                device_map={"model": "cpu", "lm_head": "cpu"},  # Load to CPU memory
                torch_dtype="auto",
            )
            tok = time.time()
            t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
            logger.info(f"HF LLaMA loaded. Total time: {t}")
            load_from_hf_llama(self._trt_model, hf_llama, mapping=mapping, dtype=self._builder_args_config.dtype)
            del hf_llama
        elif self._builder_args_config.ft_model_dir is not None:
            load_from_binary(
                self._trt_model,
                self._builder_args_config.ft_model_dir,
                mapping,
                fp16=(self._builder_args_config.dtype == "float16"),
                multi_query_mode=(self._builder_args_config.n_kv_head != self._builder_args_config.n_head),
            )

    def _get_builder_config(self, cache=None) -> tensorrt_llm.builder.BuilderConfig:
        # NOTE: when only int8 kv cache is used together with paged kv cache no int8 tensors are exposed to TRT
        int8_trt_flag = self._builder_args_config.quant_mode.has_act_and_weight_quant() or (
            not self._builder_args_config.paged_kv_cache and self._builder_args_config.quant_mode.has_int8_kv_cache()
        )
        builder_config = self._builder.create_builder_config(
            name=MODEL_NAME,
            precision=self._builder_args_config.dtype,
            timing_cache=self._builder_args_config.timing_cache if cache is None else cache,
            tensor_parallel=self._builder_args_config.tp_size,
            pipeline_parallel=self._builder_args_config.pp_size,
            parallel_build=self._builder_args_config.parallel_build,
            num_layers=self._builder_args_config.n_layer,
            num_heads=self._builder_args_config.n_head,
            num_kv_heads=self._builder_args_config.n_kv_head,
            hidden_size=self._builder_args_config.n_embd,
            vocab_size=self._builder_args_config.vocab_size,
            hidden_act=self._builder_args_config.hidden_act,
            max_position_embeddings=self._builder_args_config.n_positions,
            max_batch_size=self._builder_args_config.max_batch_size,
            max_input_len=self._builder_args_config.max_input_len,
            max_output_len=self._builder_args_config.max_output_len,
            max_num_tokens=self._builder_args_config.max_num_tokens,
            int8=int8_trt_flag,
            fp8=self._builder_args_config.quant_mode.has_fp8_qdq(),
            quant_mode=self._builder_args_config.quant_mode,
            strongly_typed=self._builder_args_config.strongly_typed,
            opt_level=self._builder_args_config.builder_opt,
        )
        return builder_config


class LlamaEngineRunner(EngineRunnerBase):
    def _parse_input(
        self, input_text: str, input_file: str, tokenizer, end_id: int, remove_input_padding: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens = []
        if input_file is None:
            input_tokens.append(tokenizer.encode(input_text, add_special_tokens=False))
        else:
            if input_file.endswith(".csv"):
                with open(input_file, "r") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    for line in csv_reader:
                        input_tokens.append(np.array(line, dtype="int32"))
            elif input_file.endswith(".npy"):
                inputs = np.load(input_file)
                for row in inputs:
                    row = row[row != end_id]
                    input_tokens.append(row)
            else:
                print("Input file format not supported.")
                raise SystemExit

        input_ids = None
        input_lengths = torch.tensor([len(x) for x in input_tokens], dtype=torch.int32, device="cuda")
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda").unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32), end_id
            ).cuda()

        return input_ids, input_lengths
