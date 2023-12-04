import json
from pathlib import Path
from typing import Tuple

import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import AutoTokenizer, LlamaTokenizer

from colossalai.inference.trtllm.utils import get_engine_name, process_output, throttle_generator


class EngineRunnerBase:
    def _read_config(self, config_path: Path, use_exist: str, config: dict) -> Tuple[ModelConfig, int, int, str]:
        if use_exist:
            assert config, "When using existing trt engine, the config must be set."
        else:
            assert config_path, "When not using existing trt engine, the config_path must be set."
            with open(config_path, "r") as f:
                config = json.load(f)
        use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
        tp_size = config["builder_config"]["tensor_parallel"]
        pp_size = config["builder_config"]["pipeline_parallel"]
        world_size = tp_size * pp_size
        assert (
            world_size == tensorrt_llm.mpi_world_size()
        ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
        num_heads = config["builder_config"]["num_heads"] // tp_size
        hidden_size = config["builder_config"]["hidden_size"] // tp_size
        vocab_size = config["builder_config"]["vocab_size"]
        num_layers = config["builder_config"]["num_layers"]
        num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)

        if config["builder_config"].get("multi_query_mode", False):
            logger.warning("`multi_query_mode` config is deprecated. Please rebuild the engine.")
            num_kv_heads = 1
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

        remove_input_padding = config["plugin_config"].get("remove_input_padding", False)
        model_name = config["plugin_config"].get("name", "")
        paged_kv_cache = config["plugin_config"].get("paged_kv_cache", False)
        cross_attention = config["builder_config"].get("cross_attention", False)
        has_position_embedding = config["builder_config"].get("has_position_embedding", True)
        has_token_type_embedding = config["builder_config"].get("has_token_type_embedding", False)
        tokens_per_block = config["plugin_config"].get("tokens_per_block", 64)
        use_prompt_tuning = config["builder_config"].get("use_prompt_tuning", False)
        quant_mode = QuantMode(getattr(config["builder_config"], "quant_mode", 0))
        gather_all_token_logits = config["builder_config"].get("gather_all_token_logits", False)
        dtype = config["builder_config"].get("precision", "")
        use_custom_all_reduce = config["plugin_config"].get("use_custom_all_reduce", False)

        model_config = ModelConfig(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            gpt_attention_plugin=use_gpt_attention_plugin,
            remove_input_padding=remove_input_padding,
            model_name=model_name,
            paged_kv_cache=paged_kv_cache,
            cross_attention=cross_attention,
            has_position_embedding=has_position_embedding,
            has_token_type_embedding=has_token_type_embedding,
            tokens_per_block=tokens_per_block,
            use_prompt_tuning=use_prompt_tuning,
            quant_mode=quant_mode,
            gather_all_token_logits=gather_all_token_logits,
            dtype=dtype,
            use_custom_all_reduce=use_custom_all_reduce,
        )

        return model_config, tp_size, pp_size, dtype

    def _parse_input(
        self, input_text: str, input_file: str, tokenizer, end_id: int, remove_input_padding: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In this interface, We need to implement the codes to process input text according to the specific model.
        """

    def _generate_decode_param(
        self,
        input_ids: torch.Tensor,
        context_lengths: torch.Tensor,
        sampling_config: SamplingConfig,
        streaming: bool = False,
    ) -> dict:
        return {
            "input_ids": input_ids,
            "context_lengths": context_lengths,
            "sampling_config": sampling_config,
            "streaming": streaming,
        }

    def generate(
        self,
        max_output_len: int,
        log_level: str = "info",
        engine_dir: str = None,
        input_text: str = "Born in north-east France, Soyer trained as a",
        input_file: str = None,
        output_csv: str = None,
        output_npy: str = None,
        tokenizer_dir: str = None,
        num_beams: int = 1,
        streaming: bool = False,
        streaming_interval: int = 5,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        debug_mode: bool = False,
        model_name: str = "",
        use_fast: bool = False,
        trust_remote_code: bool = False,
        encoder_max_input_length: int = None,
        byte_engine: bytearray = None,
        runner_config: dict = None,
        rank: int = 0,
    ) -> tensorrt_llm.runtime.GenerationSession:
        logger.set_level(log_level)
        if byte_engine and runner_config:
            use_exist = True
        else:
            use_exist = False
            assert engine_dir, "When not using existing trt engine, config_dir must be set."

        config_path = None
        if not use_exist:
            engine_dir = Path(engine_dir)
            config_path = engine_dir / "config.json"
        model_config, tp_size, pp_size, dtype = self._read_config(config_path, use_exist, runner_config)
        world_size = tp_size * pp_size
        runtime_mapping = Mapping(world_size, rank, tp_size=tp_size, pp_size=pp_size)

        torch.cuda.set_device(rank % runtime_mapping.gpus_per_node)

        if model_name == "llama":
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir, use_fast=use_fast, trust_remote_code=trust_remote_code
            )

        eos_token = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
        pad_token = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[0]

        sampling_config = SamplingConfig(
            end_id=eos_token, pad_id=pad_token, num_beams=num_beams, temperature=temperature, top_k=top_k, top_p=top_p
        )

        if not use_exist:
            engine_name = get_engine_name(model_name, dtype, tp_size, pp_size, rank)
            serialize_path = engine_dir / engine_name
            with open(serialize_path, "rb") as f:
                engine_buffer = f.read()
        else:
            engine_buffer = byte_engine

        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=debug_mode
        )

        if rank == 0:
            print(f"Running the {dtype} engine ...")

        input_ids, input_lengths = self._parse_input(
            input_text, input_file, tokenizer, eos_token, model_config.remove_input_padding
        )

        max_input_length = torch.max(input_lengths).item()
        decoder.setup(input_lengths.size(0), max_input_length, max_output_len, num_beams, encoder_max_input_length)

        if decoder == None:
            raise ValueError(f"TensorRT decode is not initialized.")

        decode_kwargs = self._generate_decode_param(input_ids, input_lengths, sampling_config, streaming=streaming)

        output_gen_ids = decoder.decode(**decode_kwargs)
        torch.cuda.synchronize()

        outputs = []

        if streaming:
            for output_ids in throttle_generator(output_gen_ids, streaming_interval):
                if rank == 0:
                    outputs.append(
                        process_output(output_ids, input_lengths, max_output_len, tokenizer, output_csv, output_npy)
                    )
        else:
            output_ids = output_gen_ids
            if rank == 0:
                outputs.append(
                    process_output(output_ids, input_lengths, max_output_len, tokenizer, output_csv, output_npy)
                )
        return outputs
